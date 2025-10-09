# backend/main.py
import os
import re
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

# Optional: OpenAI (only used if OPENAI_API_KEY set)
try:
    import openai
except Exception:
    openai = None

# Load small English model (install instructions below)
nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Cross-Culture Humor Mapper - Prototype")


class AdaptRequest(BaseModel):
    text: str
    source_culture: str = "US_en"
    target_culture: str = "IN_en"


# small local analog dictionary for prototype (expand as needed)
LOCAL_ANALOGS = {
    # (entity_lower, source_culture) -> list of candidates for target culture
    ("donald trump", "US_en", "IN_en"): ["Narendra Modi", "Amit Shah"],
    ("barack obama", "US_en", "IN_en"): ["Dr. Manmohan Singh", "Narendra Modi"],
    ("mcdonald's", "US_en", "IN_en"): ["Domino's", "Haldiram's"],
    ("late-night host", "US_en", "IN_en"): ["a popular Indian TV host"]
}


def simple_humor_type(joke: str) -> List[str]:
    """Very simple rule-based humor type detector."""
    jok = joke.lower().strip()
    types = []
    if re.search(r"\bwhy\b.*\?", jok) or jok.endswith("?"):
        types.append("q-and-a")
    if re.search(r"\bpun\b|\bwordplay\b", jok) or re.search(r"\b(?:(looked surprised|looked shocked))\b", jok):
        types.append("wordplay")
    # obvious pun-ish pattern: short setup + punchline with unexpected phrase
    if len(jok.split()) < 20 and ("because" in jok or jok.count(",") >= 1):
        types.append("one-liner")
    # fallback
    if not types:
        types = ["observational"]
    return list(dict.fromkeys(types))


def extract_entities(joke: str) -> List[Dict[str, Any]]:
    doc = nlp(joke)
    ents = []
    for ent in doc.ents:
        ents.append({"text": ent.text, "label": ent.label_})
    return ents


def map_entities(entities: List[Dict[str, Any]], source_culture: str, target_culture: str) -> Dict[str, List[str]]:
    """
    Prototype mapping: uses LOCAL_ANALOGS for demo.
    Returns a mapping {original_entity_lower: [candidates]}
    """
    result = {}
    for e in entities:
        key = (e["text"].lower(), source_culture, target_culture)
        candidates = LOCAL_ANALOGS.get(key)
        if candidates:
            result[e["text"]] = candidates
    return result


def rule_based_rewrite(joke: str, mappings: Dict[str, List[str]], target_culture: str) -> Dict[str, str]:
    """
    Simple deterministic rewrite: replace first mappable entity with top candidate.
    If nothing to map, try to preserve structure and replace a celebrity with 'a local celebrity'.
    """
    adapted = joke
    explanation = []
    if mappings:
        for orig, cands in mappings.items():
            adapted = re.sub(re.escape(orig), cands[0], adapted, flags=re.IGNORECASE)
            explanation.append(f"Replaced '{orig}' with '{cands[0]}' for {target_culture}.")
    else:
        # heuristic: replace "American" references
        adapted = re.sub(r"\bAmerican\b", "local", adapted, flags=re.IGNORECASE)
        explanation.append("No direct analog found: neutralized nationality references.")
    return {"adapted": adapted, "explanation": " ".join(explanation)}


def build_prompt(joke: str, analysis: dict, mappings: Dict[str, List[str]], source_culture: str, target_culture: str):
    # compact explanation-based prompt
    candidate_text = ""
    for orig, cands in mappings.items():
        candidate_text += f"- {orig} -> {', '.join(cands)}\n"
    prompt = f"""You are a comedy writer adapting jokes across cultures.
Source culture: {source_culture}
Target culture: {target_culture}
Joke: "{joke}"

Analysis:
{json.dumps(analysis)}

Entity mapping candidates:
{candidate_text if candidate_text else 'None'}

Task:
Provide a single best adaptation of the joke for the target culture that preserves the punchline or equivalent effect. Then write a one-sentence explanation of what you changed and why. If you think the joke cannot be adapted while preserving the punchline, say so and offer a neutralized version.

Return just JSON with keys: adapted, explanation.
"""
    return prompt


def call_openai_chat(prompt: str) -> Dict[str, str]:
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI or API key not available.")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Use ChatCompletion or Chat API - keep it simple
    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[
            {"role": "system", "content": "You adapt jokes for target cultures with sensitivity."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    txt = resp["choices"][0]["message"]["content"]
    # expect a JSON; try to parse, otherwise wrap in fields
    try:
        parsed = json.loads(txt)
        return parsed
    except Exception:
        # fallback: return raw text in explanation
        return {"adapted": txt.strip(), "explanation": "Adaptation returned as text; could not parse JSON."}


@app.post("/adapt")
def adapt(req: AdaptRequest):
    text = req.text.strip()
    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Empty text.")
    # 1. Analysis
    ents = extract_entities(text)
    humor_types = simple_humor_type(text)
    analysis = {"entities": ents, "humor_types": humor_types}

    # 2. Mapping
    mappings = map_entities(ents, req.source_culture, req.target_culture)

    # 3. Rewrite: prefer OpenAI if key present, else rule-based fallback
    try:
        if os.getenv("OPENAI_API_KEY") and openai is not None:
            prompt = build_prompt(text, analysis, mappings, req.source_culture, req.target_culture)
            llm_out = call_openai_chat(prompt)
            adapted = llm_out.get("adapted", "").strip()
            explanation = llm_out.get("explanation", "")
            source = "openai"
        else:
            fallback = rule_based_rewrite(text, mappings, req.target_culture)
            adapted = fallback["adapted"]
            explanation = fallback["explanation"]
            source = "rule-based"
    except Exception as e:
        # on any error, fallback gracefully
        fallback = rule_based_rewrite(text, mappings, req.target_culture)
        adapted = fallback["adapted"]
        explanation = f"Fallback due to error: {str(e)}. " + fallback["explanation"]
        source = "rule-based-fallback"

    return {
        "source_text": text,
        "analysis": analysis,
        "mappings": mappings,
        "adaptation": {
            "text": adapted,
            "explanation": explanation,
            "engine": source
        }
    }


@app.get("/health")
def health():
    return {"ok": True}
