"""
Hybrid NLP + Linguistic Pattern Analyzer for Common IELTS Errors.

Combines a trained TF-IDF (word + character n-grams) Logistic Regression classifier
with interpretable domain-specific regex signals common among Vietnamese IELTS learners.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

from ml.data_generator import ERROR_CATEGORIES

# Lightweight linguistic indicators that boost precision on unseen real-world sentences
LINGUISTIC_RULES: Dict[str, List[Tuple[str, str]]] = {
    "grammar_tense": [
        (r"\b(the number of .+? (have|are|were))\b", "'the number of + N' cần động từ chia số ít"),
        (r"\b(he|she|it|everyone|everybody|each)\s+(go|do|have|work|study|make|need|want|play)\b", "Thiếu chia động từ ngôi thứ 3 số ít"),
        (r"\b(he|she|it|student|government|technology)\s+don't\b", "Dùng 'don't' thay vì 'doesn't' cho chủ ngữ số ít"),
        (r"\b(didn't|did not)\s+(went|saw|had|made|did|came|bought|wrote)\b", "Sau 'didn't' cần động từ nguyên mẫu (V0)"),
        (r"\bthere\s+is\s+(many|several|a lot of|two|three|some)\b", "'There is' không khớp với danh từ số nhiều"),
        (r"\b(last year|yesterday|in 19\d\d|in 20[01]\d).*\b(have|has)\s+\w+ed\b", "Không dùng Hiện tại hoàn thành với mốc quá khứ xác định"),
        (r"\b(people|students|children)\s+was\b", "Chủ ngữ số nhiều cần 'were' thay vì 'was'"),
        (r"\bwill\s+(grew|went|had|became|increased|decreased)\b", "Sau 'will' cần động từ nguyên mẫu (V0)"),
        (r"(chia sai thì|hòa hợp chủ|quá khứ đơn|hiện tại hoàn thành|thêm s/es)", "Dấu hiệu mô tả lỗi thì / hòa hợp chủ-vị"),
    ],
    "article_preposition": [
        (r"\b(discuss|mention|emphasize|approach|reach)\s+about\b", "Động từ này không đi kèm giới từ 'about'"),
        (r"\bdespite\s+of\b", "Dùng 'Despite + N/V-ing' hoặc 'In spite of', không dùng 'Despite of'"),
        (r"\bdepend\s+in\b", "Cụm đúng là 'depend on', không phải 'depend in'"),
        (r"\binterested\s+on\b", "Cụm đúng là 'interested in'"),
        (r"\bsuffer\s+of\b", "Cụm đúng là 'suffer from'"),
        (r"\bmarried\s+with\b", "Cụm đúng là 'married to'"),
        (r"\baccording\s+to\s+my\s+opinion\b", "Không dùng 'according to my opinion' → dùng 'in my opinion'"),
        (r"\b(in)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "Trước thứ trong tuần dùng giới từ 'on'"),
        (r"\b(a)\s+(honest|hour|umbrella|university student)?\b.*\b(an university|a honest)\b", "Sai lựa chọn mạo từ a/an theo âm đầu"),
        (r"\b(an university|a honest|in the vietnam|in the china|in the japan)\b", "Sai quy tắc dùng mạo từ a/an/the"),
        (r"\bpay attention for\b|\bconsists from\b|\bcompared by\b|\barrived to\b", "Sai giới từ cố định đi kèm động từ"),
        (r"(mạo từ|giới từ|a/an/the|thiếu the|thừa giới từ)", "Dấu hiệu mô tả lỗi mạo từ / giới từ"),
    ],
    "collocation_vocab": [
        (r"\b(do|did|doing)\s+(a\s+)?(mistake|mistakes|progress|harm for)\b", "Collocation sai: dùng 'make a mistake / make progress / do harm to'"),
        (r"\b(informations|advices|equipments|knowledges|vocabularies|homeworks)\b", "Danh từ không đếm được không thêm 's'"),
        (r"\b(big rain|open the light|very like|make sports|take a good salary|said me)\b", "Dịch word-by-word không tự nhiên (collocation sai)"),
        (r"\bhealthy problems\b|\brise the risk\b|\bmake a big influence\b", "Dùng sai kết hợp từ (collocation)"),
        (r"(collocation|word-by-word|không đếm được|dùng từ không tự nhiên|lặp từ vựng)", "Dấu hiệu mô tả lỗi collocation / từ vựng"),
    ],
    "sentence_structure": [
        (r"\b(although|though|even though)\b.{3,80}\b(but|however|yet)\b", "Lỗi dùng kép liên từ nhượng bộ (Although... but/however)"),
        (r"\bbecause\b.{3,80}\bso\b", "Lỗi dùng kép liên từ nguyên nhân (Because... so)"),
        (r",\s*(however|therefore|moreover|furthermore)\s+[a-z]", "Lỗi comma splice khi nối 2 mệnh đề độc lập với trạng từ nối (cần dấu chấm phẩy ';' hoặc chấm '.')"),
        (r"\bthere\s+(are|is)\s+many\s+\w+\s+(want|like|use|think|need|live)\b", "Thiếu đại từ quan hệ 'who/that' trong câu có 'There are...'"),
        (r"\bdespite\s+(he|she|they|we|it|the\s+\w+\s+(is|are|was|were))\b", "Sau 'Despite' phải là cụm danh từ/V-ing, không dùng mệnh đề S+V"),
        (r"\bbecause\s+of\s+.{2,40}\b(is|are|was|were)\b", "Sau 'Because of' phải là cụm danh từ, không dùng mệnh đề S+V"),
        (r"\bwhile\b.{3,60}\bwhereas\b", "Dùng thừa hai liên từ đối lập (while... whereas)"),
        (r"(run-on|comma splice|fragment|câu cụt|kép liên từ|mệnh đề quan hệ)", "Dấu hiệu mô tả lỗi cấu trúc câu"),
    ],
    "coherence_cohesion": [
        (r"\bin the other hand\b", "Cụm nối chuẩn là 'On the other hand'"),
        (r"(firstly.*secondly.*thirdly|however.*however|moreover.*furthermore.*in addition)", "Lạm dụng từ nối máy móc lặp lại"),
        (r"\b(every coin has two sides|last but not least|by the way)\b", "Cụm nối sáo rỗng hoặc không phù hợp văn phong học thuật"),
        (r"(topic sentence|câu chủ đề|liên kết|mạch lạc|nhảy cóc|từ nối máy móc|đại từ thay thế)", "Dấu hiệu mô tả lỗi liên kết & mạch lạc"),
    ],
    "task_response": [
        (r"\b(under 150|under 250|only wrote \d+ words|off-topic|no overview|no opinion|both views)\b", "Vi phạm yêu cầu đề bài Task Response / Task Achievement"),
        (r"(lạc đề|thiếu overview|thiếu số từ|không nêu rõ quan điểm|hai vế|lập luận hời hợt|chủ quan trong task 1)", "Dấu hiệu mô tả lỗi Task Response"),
    ],
    "spelling_wordform": [
        (r"\b(goverment|enviroment|recieve|accomodation|neccessary|seperate|definetly|imediate|oppurtunities|conveniant|explaination|polution|deseases)\b", "Sai chính tả từ học thuật phổ biến"),
        (r"\bmore\s+(easier|better|cheaper|faster|harder|higher|bigger)\b", "Lỗi so sánh hơn kép (không dùng 'more' trước tính từ đuôi -er)"),
        (r"\b(very benefit|very intelligence|very successfully|more safety|big different|be educate|worked very hardly|culture exchanges|finance independence|danger situation)\b", "Dùng sai từ loại (danh/tính/trạng từ)"),
        (r"(chính tả|từ loại|word form|so sánh hơn kép|đuôi -ed và -ing)", "Dấu hiệu mô tả lỗi chính tả / từ loại"),
    ],
    "listening_reading_trap": [
        (r"\b(no more than|not given|true/false|distractor|self-correction|paraphrase|plural|word limit|postcode|map labelling|heading)\b", "Bẫy phổ biến trong bài thi Listening / Reading"),
        (r"(bẫy|distractor|not given|giới hạn từ|số nhiều|paraphrase|từ đồng nghĩa|bản đồ|nghe|đọc)", "Dấu hiệu mô tả lỗi Listening / Reading"),
    ],
}


def extract_rule_signals(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Scan text against curated linguistic rules to produce interpretable boosts & matched hints."""
    lowered = text.lower().strip()
    scores: Dict[str, float] = {cat: 0.0 for cat in ERROR_CATEGORIES}
    reasons: Dict[str, List[str]] = {cat: [] for cat in ERROR_CATEGORIES}

    for category, patterns in LINGUISTIC_RULES.items():
        for pattern, explanation in patterns:
            if re.search(pattern, lowered):
                scores[category] += 1.0
                if explanation not in reasons[category]:
                    reasons[category].append(explanation)

    return scores, reasons


def split_into_segments(raw_text: str) -> List[str]:
    """Split an essay or error log into clean sentences/entries for granular classification."""
    lines = [line.strip(" -•*\t") for line in raw_text.splitlines() if line.strip()]
    segments: List[str] = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) >= 6:
                segments.append(cleaned)
    return segments if segments else ([raw_text.strip()] if raw_text.strip() else [])


def classify_single_error(
    text: str,
    nlp_pipeline: Any,
    classes: List[str],
) -> Dict[str, Any]:
    """
    Predict the most likely IELTS error category for a single text snippet using
    hybrid ML probabilities (TF-IDF + LogisticRegression) + linguistic rule signals.
    """
    cleaned = text.strip()
    ml_probs = nlp_pipeline.predict_proba([cleaned])[0]
    prob_map = {cls: float(p) for cls, p in zip(classes, ml_probs)}

    rule_scores, rule_reasons = extract_rule_signals(cleaned)
    total_rule_hits = sum(rule_scores.values())

    # Blend ML probability (70%) with linguistic rule evidence (30%) when rules match
    blended: Dict[str, float] = {}
    if total_rule_hits > 0:
        for cls in classes:
            rule_prob = rule_scores.get(cls, 0.0) / total_rule_hits
            blended[cls] = 0.65 * prob_map[cls] + 0.35 * rule_prob
    else:
        blended = dict(prob_map)

    # Normalize to sum to 1
    norm = sum(blended.values()) or 1.0
    blended = {k: round(v / norm, 4) for k, v in blended.items()}

    sorted_cats = sorted(blended.items(), key=lambda item: item[1], reverse=True)
    top_cat, top_conf = sorted_cats[0]
    meta = ERROR_CATEGORIES[top_cat]

    return {
        "text": cleaned,
        "category": top_cat,
        "label_vi": meta["label_vi"],
        "label_en": meta["label_en"],
        "skill": meta["skill"],
        "confidence": round(top_conf * 100.0, 1),
        "description_vi": meta["description_vi"],
        "advice_vi": meta["advice_vi"],
        "example_fix": meta["example_fix"],
        "detected_signals": rule_reasons.get(top_cat, []),
        "top_probabilities": [
            {
                "category": cat,
                "label_vi": ERROR_CATEGORIES[cat]["label_vi"],
                "probability": round(prob * 100.0, 1),
            }
            for cat, prob in sorted_cats[:4]
        ],
    }


def analyze_error_batch(
    raw_text: str,
    nlp_pipeline: Any,
    classes: List[str],
) -> Dict[str, Any]:
    """Classify every sentence/line in a text block and summarize the most common errors."""
    segments = split_into_segments(raw_text)
    if not segments:
        return {
            "total_segments": 0,
            "dominant_error": None,
            "category_distribution": [],
            "segment_results": [],
            "remediation_plan": [],
        }

    results = [classify_single_error(seg, nlp_pipeline, classes) for seg in segments]

    counts: Dict[str, int] = {}
    conf_sums: Dict[str, float] = {}
    for res in results:
        cat = res["category"]
        counts[cat] = counts.get(cat, 0) + 1
        conf_sums[cat] = conf_sums.get(cat, 0.0) + res["confidence"]

    total = len(results)
    distribution = []
    for cat, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], -conf_sums[kv[0]])):
        meta = ERROR_CATEGORIES[cat]
        distribution.append(
            {
                "category": cat,
                "label_vi": meta["label_vi"],
                "label_en": meta["label_en"],
                "skill": meta["skill"],
                "count": cnt,
                "percentage": round(cnt / total * 100.0, 1),
                "avg_confidence": round(conf_sums[cat] / cnt, 1),
                "advice_vi": meta["advice_vi"],
                "example_fix": meta["example_fix"],
            }
        )

    dominant = distribution[0]
    remediation_plan = [
        {
            "priority": i + 1,
            "error_type": item["label_vi"],
            "share": f"{item['percentage']}% ({item['count']}/{total} câu)",
            "action": item["advice_vi"],
            "example": item["example_fix"],
        }
        for i, item in enumerate(distribution[:3])
    ]

    return {
        "total_segments": total,
        "dominant_error": dominant,
        "category_distribution": distribution,
        "segment_results": results,
        "remediation_plan": remediation_plan,
    }

