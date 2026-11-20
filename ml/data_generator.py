"""
Synthetic yet realistic dataset generator for IELTS Learning Analytics.

Generates two CSV datasets:
1. data/ielts_students.csv: Student entry score, WOW coaching sessions, attendance %,
   homework completion %, 4-skill homework % scores (L/R/W/S), learning quality score,
   actual IELTS exam bands (L/R/W/S/Overall), and strength/weakness tier.
2. data/ielts_errors.csv: Labeled IELTS learner errors (English sentences & teacher feedback notes)
   across 8 common error categories for Vietnamese IELTS learners.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

STUDENT_COLUMNS = [
    "student_id",
    "entry_score",
    "wow_sessions",
    "attendance_rate",
    "homework_completion_rate",
    "hw_listening_pct",
    "hw_reading_pct",
    "hw_writing_pct",
    "hw_speaking_pct",
    "learning_quality_score",
    "actual_listening",
    "actual_reading",
    "actual_writing",
    "actual_speaking",
    "actual_overall",
    "student_tier",
]

ERROR_CATEGORIES: Dict[str, Dict[str, str]] = {
    "grammar_tense": {
        "label_vi": "Lỗi Thì & Hòa hợp Chủ-Vị",
        "label_en": "Tense & Subject-Verb Agreement",
        "skill": "Writing / Speaking",
        "description_vi": "Chia sai thì động từ (quá khứ/hiện tại/tương lai), quên thêm s/es cho ngôi thứ 3 số ít, hoặc không thống nhất thì trong đoạn văn.",
        "advice_vi": "Luôn xác định mốc thời gian của câu trước khi chia động từ. Kiểm tra lại chủ ngữ số ít/số nhiều ở bước soát bài (proofreading) cuối cùng.",
        "example_fix": "Sai: 'The number of students have increased last year.' → Đúng: 'The number of students increased last year.'",
    },
    "article_preposition": {
        "label_vi": "Lỗi Mạo từ & Giới từ",
        "label_en": "Articles & Prepositions",
        "skill": "Writing / Speaking",
        "description_vi": "Thiếu hoặc dùng thừa mạo từ (a/an/the), dùng sai giới từ đi kèm động từ/danh từ (ví dụ: discuss about, depend in, in Monday).",
        "advice_vi": "Học danh từ kèm mạo từ và động từ kèm giới từ cố định (verb + preposition). Lập danh sách 30 cụm giới từ hay sai nhất của người Việt.",
        "example_fix": "Sai: 'We should discuss about the problem in detail.' → Đúng: 'We should discuss the problem in detail.'",
    },
    "collocation_vocab": {
        "label_vi": "Lỗi Collocation & Dùng từ sai ngữ cảnh",
        "label_en": "Collocations & Word Choice",
        "skill": "Writing / Speaking",
        "description_vi": "Dịch word-by-word từ tiếng Việt sang tiếng Anh, dùng từ không tự nhiên (do a mistake, big rain, open the light) hoặc danh từ không đếm được dùng số nhiều (informations, advices).",
        "advice_vi": "Học từ vựng theo cụm (collocations) trong Oxford Collocations Dictionary thay vì học từ đơn lẻ. Ghi nhớ danh sách danh từ không đếm được phổ biến.",
        "example_fix": "Sai: 'Many students do mistakes when learning English.' → Đúng: 'Many students make mistakes when learning English.'",
    },
    "sentence_structure": {
        "label_vi": "Lỗi Cấu trúc câu & Mệnh đề",
        "label_en": "Sentence Structure & Clauses",
        "skill": "Writing",
        "description_vi": "Câu thiếu động từ chính (fragment), nối hai mệnh đề độc lập chỉ bằng dấu phẩy (comma splice / run-on), hoặc dùng kép liên từ (Although... but, Because... so).",
        "advice_vi": "Mỗi câu đơn phải có đủ Chủ ngữ + Động từ chính chia thì. Tuyệt đối không dùng đồng thời Although với But, hoặc Because với So trong cùng một câu.",
        "example_fix": "Sai: 'Although it rained heavily, but they went to school.' → Đúng: 'Although it rained heavily, they went to school.'",
    },
    "coherence_cohesion": {
        "label_vi": "Lỗi Liên kết & Mạch lạc đoạn văn",
        "label_en": "Coherence & Cohesion",
        "skill": "Writing / Speaking",
        "description_vi": "Lạm dụng từ nối máy móc (Firstly, Moreover ở mọi câu), thiếu câu chủ đề (topic sentence), ý tưởng nhảy cóc không logic hoặc đại từ thay thế (it/they/this) gây mơ hồ.",
        "advice_vi": "Áp dụng cấu trúc PEEL (Point - Explain - Example - Link) cho mỗi đoạn thân bài. Dùng phép quy chiếu tự nhiên thay vì nhồi nhét linking words.",
        "example_fix": "Sai: 'In the other hand, technology is useful. It helps them.' → Đúng: 'On the other hand, technology is useful because it helps students access knowledge.'",
    },
    "task_response": {
        "label_vi": "Lỗi Lạc đề & Thiếu lập luận Task Response",
        "label_en": "Task Response / Achievement",
        "skill": "Writing",
        "description_vi": "Trả lời thiếu một vế câu hỏi, không nêu rõ quan điểm cá nhân (opinion), đưa ví dụ quá chung chung/sáo rỗng, hoặc viết dưới số từ quy định (150/250 từ).",
        "advice_vi": "Dành 5 phút đầu phân tích kỹ đề bài (keywords, dạng bài, số lượng câu hỏi). Đảm bảo lập luận đi sâu phân tích nguyên nhân - hệ quả thay vì liệt kê bề mặt.",
        "example_fix": "Thiếu quan điểm: Nên nêu rõ thesis statement ở Mở bài và khẳng định lại nhất quán ở Kết bài.",
    },
    "spelling_wordform": {
        "label_vi": "Lỗi Chính tả & Từ loại (Word Form)",
        "label_en": "Spelling & Word Form",
        "skill": "Writing / Listening",
        "description_vi": "Viết sai chính tả các từ học thuật phổ biến (goverment, enviroment, recieve) hoặc nhầm lẫn danh từ / tính từ / trạng từ (success vs successful, economic vs economy).",
        "advice_vi": "Kiểm tra chức năng ngữ pháp của vị trí từ cần điền (sau to be dùng tính từ, bổ nghĩa động từ dùng trạng từ). Ôn tập danh sách 100 từ dễ viết sai chính tả trong IELTS Listening & Writing.",
        "example_fix": "Sai: 'This policy is very benefit for the enviroment.' → Đúng: 'This policy is very beneficial for the environment.'",
    },
    "listening_reading_trap": {
        "label_vi": "Lỗi Bẫy Paraphrase & Distractor Nghe-Đọc",
        "label_en": "Listening/Reading Traps & Paraphrasing",
        "skill": "Listening / Reading",
        "description_vi": "Bị đánh lừa bởi thông tin gây nhiễu (speaker tự sửa lại thông tin), nhầm lẫn giữa FALSE và NOT GIVEN trong Reading, viết vượt quá giới hạn từ (NO MORE THAN TWO WORDS), hoặc bỏ sót số nhiều/đơn vị.",
        "advice_vi": "Gạch chân giới hạn từ (word limit) trước khi làm bài. Trong Listening luôn lắng nghe đến hết câu vì người nói hay đổi ý sau 'actually/sorry/wait'. Trong Reading phân biệt rõ: mâu thuẫn = FALSE, không đề cập = NOT GIVEN.",
        "example_fix": "Trap: 'Let's meet on Tuesday... oh sorry, I'm busy, let's make it Thursday.' → Đáp án đúng là Thursday, không phải Tuesday.",
    },
}


def round_ielts_band(score: float) -> float:
    """Official IELTS band rounding rule: nearest 0.5 (.25 rounds up to .5, .75 rounds up to next integer)."""
    clamped = max(1.0, min(9.0, float(score)))
    return round(clamped * 2.0 + 1e-9) / 2.0


def compute_official_overall(l_band: float, r_band: float, w_band: float, s_band: float) -> float:
    """Official IELTS Overall Band is the average of 4 skills rounded to the nearest 0.5."""
    avg = (l_band + r_band + w_band + s_band) / 4.0
    frac = avg - np.floor(avg)
    if frac < 0.25:
        return float(np.floor(avg))
    if frac < 0.75:
        return float(np.floor(avg) + 0.5)
    return float(min(9.0, np.floor(avg) + 1.0))


def classify_student_tier(overall_band: float) -> str:
    """Map IELTS actual overall score into 4 pedagogical strength/weakness tiers."""
    if overall_band < 5.0:
        return "Yếu"
    if overall_band < 6.0:
        return "Trung bình"
    if overall_band < 7.0:
        return "Khá"
    return "Mạnh"


def generate_student_dataset(n_samples: int = 1800, seed: int = 42) -> List[Dict[str, object]]:
    """
    Generate realistic synthetic IELTS center student records with exact input fields:
    - entry_score (Điểm đầu vào: 2.5 - 7.5)
    - wow_sessions (Số buổi đặt WOW: 0 - 30)
    - attendance_rate (% chuyên cần: 20 - 100)
    - homework_completion_rate (% làm bài tập về nhà: 15 - 100)
    - hw_listening_pct, hw_reading_pct, hw_writing_pct, hw_speaking_pct (% điểm 4 kỹ năng BTVN)
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    # 4 realistic learner archetypes
    # 0: Yếu / ít tham gia, 1: Trung bình / thiếu đều, 2: Khá / chăm chỉ, 3: Mạnh / xuất sắc
    archetypes = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.22, 0.28, 0.30, 0.20])

    for idx in range(n_samples):
        arch = int(archetypes[idx])

        if arch == 0:
            entry = round_ielts_band(rng.normal(3.8, 0.55))
            wow_sessions = int(np.clip(rng.normal(3.0, 2.2), 0, 12))
            attendance = float(np.clip(rng.normal(62.0, 12.0), 20.0, 100.0))
            homework = float(np.clip(rng.normal(50.0, 14.0), 15.0, 100.0))
            base_hw_pct = rng.normal(46.0, 9.0)
        elif arch == 1:
            entry = round_ielts_band(rng.normal(4.8, 0.55))
            wow_sessions = int(np.clip(rng.normal(8.0, 3.2), 0, 20))
            attendance = float(np.clip(rng.normal(76.0, 10.0), 30.0, 100.0))
            homework = float(np.clip(rng.normal(68.0, 11.0), 25.0, 100.0))
            base_hw_pct = rng.normal(62.0, 8.5)
        elif arch == 2:
            entry = round_ielts_band(rng.normal(5.6, 0.50))
            wow_sessions = int(np.clip(rng.normal(14.0, 3.8), 2, 26))
            attendance = float(np.clip(rng.normal(88.0, 7.0), 50.0, 100.0))
            homework = float(np.clip(rng.normal(84.0, 8.0), 45.0, 100.0))
            base_hw_pct = rng.normal(75.0, 7.5)
        else:
            entry = round_ielts_band(rng.normal(6.5, 0.55))
            wow_sessions = int(np.clip(rng.normal(20.0, 4.5), 6, 32))
            attendance = float(np.clip(rng.normal(95.0, 4.0), 70.0, 100.0))
            homework = float(np.clip(rng.normal(93.0, 5.0), 65.0, 100.0))
            base_hw_pct = rng.normal(87.0, 6.0)

        entry = float(np.clip(entry, 2.0, 8.0))

        # 4-skill homework percentages (Vietnamese learners often slightly higher in L/R than W/S)
        hw_l = float(np.clip(base_hw_pct + rng.normal(3.5, 7.0), 15.0, 100.0))
        hw_r = float(np.clip(base_hw_pct + rng.normal(4.0, 7.0), 15.0, 100.0))
        hw_w = float(np.clip(base_hw_pct + rng.normal(-4.5, 6.5), 15.0, 100.0))
        hw_s = float(np.clip(base_hw_pct + rng.normal(-3.0, 7.0), 15.0, 100.0))
        avg_hw_pct = (hw_l + hw_r + hw_w + hw_s) / 4.0

        # Learning Quality Score (0 - 100)
        wow_norm = min(wow_sessions / 18.0, 1.0) * 100.0
        quality_raw = (
            0.26 * attendance
            + 0.26 * homework
            + 0.20 * wow_norm
            + 0.28 * avg_hw_pct
            + rng.normal(0.0, 2.2)
        )
        learning_quality_score = float(np.clip(round(quality_raw, 1), 10.0, 99.5))

        # Actual IELTS Band per skill driven by entry_score + homework skill % + attendance/HW/WOW uplift
        effort_boost = (
            0.45 * ((attendance - 70.0) / 30.0)
            + 0.45 * ((homework - 70.0) / 30.0)
            + 0.55 * min(wow_sessions / 18.0, 1.15)
        )

        def simulate_actual_skill(hw_pct: float, wow_weight: float, skill_bias: float) -> float:
            # Map homework % (20..100) smoothly into IELTS band scale (~3.0..8.7)
            hw_band_signal = 1.6 + (hw_pct / 100.0) * 7.1
            blended_base = 0.44 * entry + 0.56 * hw_band_signal
            headroom = max(0.25, (9.0 - blended_base) / 3.5)
            wow_extra = min(wow_sessions / 20.0, 1.0) * wow_weight
            raw_band = (
                blended_base
                + (effort_boost * 0.45 + wow_extra) * headroom
                + skill_bias
                + rng.normal(0.0, 0.18)
            )
            return round_ielts_band(raw_band)

        # WOW coaching sessions have strongest extra impact on Speaking & Writing
        actual_l = simulate_actual_skill(hw_l, wow_weight=0.20, skill_bias=0.05)
        actual_r = simulate_actual_skill(hw_r, wow_weight=0.20, skill_bias=0.05)
        actual_w = simulate_actual_skill(hw_w, wow_weight=0.45, skill_bias=-0.08)
        actual_s = simulate_actual_skill(hw_s, wow_weight=0.50, skill_bias=-0.04)
        actual_overall = compute_official_overall(actual_l, actual_r, actual_w, actual_s)
        tier = classify_student_tier(actual_overall)

        rows.append(
            {
                "student_id": f"STU-{idx + 1:04d}",
                "entry_score": entry,
                "wow_sessions": wow_sessions,
                "attendance_rate": round(attendance, 1),
                "homework_completion_rate": round(homework, 1),
                "hw_listening_pct": round(hw_l, 1),
                "hw_reading_pct": round(hw_r, 1),
                "hw_writing_pct": round(hw_w, 1),
                "hw_speaking_pct": round(hw_s, 1),
                "learning_quality_score": learning_quality_score,
                "actual_listening": actual_l,
                "actual_reading": actual_r,
                "actual_writing": actual_w,
                "actual_speaking": actual_s,
                "actual_overall": actual_overall,
                "student_tier": tier,
            }
        )

    return rows


def _base_error_examples() -> Dict[str, List[str]]:
    """Curated realistic English learner sentences and teacher notes per IELTS error category."""
    return {
        "grammar_tense": [
            "The number of university students have increased sharply last year.",
            "In 2010, the percentage of people using cars rises to 60 percent.",
            "She go to the library every day to prepare for her IELTS exam.",
            "Yesterday I have visited my hometown and see many changes.",
            "The government don't invest enough money in public transport.",
            "People was working longer hours in the past decade.",
            "Everyone in my class are excited about the upcoming test.",
            "Between 2000 and 2015, carbon emissions was increasing rapidly.",
            "Technology have changed the way people communicate since 1990.",
            "He has finish his essay before the deadline yesterday.",
            "Nowadays, many young people preferred watching videos to reading books.",
            "The chart show that oil production decrease from 2005 to 2010.",
            "My brother work as an engineer and live in Hanoi.",
            "If the government will ban cars, air pollution decreases.",
            "Students who studies hard usually gets high band scores.",
            "There is many reasons why unemployment rose in 2020.",
            "There are a significant drop in birth rates last century.",
            "By 2050, the population will grew to ten billion people.",
            "She didn't went to school because she was sick.",
            "The data clearly indicate that sales was highest in July.",
            "Lỗi chia sai thì quá khứ đơn trong IELTS Writing Task 1 biểu đồ năm 2015.",
            "Sai sự hòa hợp chủ ngữ và động từ số ít số nhiều, quên thêm s/es.",
            "Dùng sai thì hiện tại hoàn thành với mốc thời gian xác định trong quá khứ.",
            "Nhầm lẫn thì hiện tại đơn và quá khứ đơn khi kể lại trải nghiệm Speaking Part 2.",
            "Động từ chia không khớp với chủ ngữ the number of và a number of.",
            "Sai bị động quá khứ was build thay vì was built.",
            "Many children doesn't like eating vegetables at school.",
            "Since 2018, online learning became more popular worldwide.",
            "The graph illustrated the changes in energy use over the next ten years.",
            "Each of the participants have to submit a report.",
        ],
        "article_preposition": [
            "We need to discuss about the impact of climate change in society.",
            "Students depend in their teachers to improve writing skills.",
            "I was born on 2002 in the Vietnam and live at Hanoi.",
            "Government should invest money for education and healthcare.",
            "In Monday morning, I have an important IELTS mock test.",
            "She is very interested on learning foreign languages.",
            "The internet plays important role in modern education.",
            "He is a honest person who works at an university.",
            "We arrived to the airport late because of heavy traffic.",
            "This solution can help to solve to the environmental crisis.",
            "Despite of the high cost, many parents send children abroad.",
            "According to my opinion, public transport should be free.",
            "Children spend too much time in playing video games.",
            "I am looking forward to hear from you in the future.",
            "The population in the China is larger than in the Japan.",
            "It is difficult for students to focus on the study at the night.",
            "Parents should pay attention for their children's mental health.",
            "Many people suffer of stress due to heavy workload.",
            "Compared by the previous year, sales increased significantly.",
            "She married with a doctor and moved to the Australia.",
            "Thiếu mạo từ a an the trước danh từ đếm được số ít.",
            "Dùng thừa giới từ about sau động từ discuss và mention.",
            "Dùng sai giới từ in on at cho thời gian và địa điểm.",
            "Sai cụm giới từ đi kèm tính từ như interested in, good at, responsible for.",
            "Thêm the sai quy tắc trước danh từ trừu tượng hoặc tên quốc gia.",
            "The society needs better rules to protect environment.",
            "I usually go to the bed early on the weekend.",
            "Students lack of practical experience after graduation.",
            "This book consists from five chapters about economics.",
            "There is an increase of the number of tourists visiting city.",
        ],
        "collocation_vocab": [
            "Many students do a lot of mistakes in the IELTS Writing test.",
            "We had a big rain yesterday so the streets were flooded.",
            "Please open the light because the room is very dark.",
            "Using social media brings many bad effects to young people.",
            "The teacher gave us many useful advices and informations.",
            "I very like watching movies in my free time.",
            "He wants to take a good salary after graduating from university.",
            "People should make sports regularly to keep healthy.",
            "Fast food can cause serious healthy problems like obesity.",
            "We need to widen our knowledge by reading news everyday.",
            "Smoking can do harm for our lungs and heart.",
            "The city has a heavy traffic jam every morning.",
            "I want to reach my dream of studying abroad.",
            "This policy will make a big influence to the economy.",
            "Students should learn by heart new vocabularies and knowledges.",
            "The company did a great progress in renewable energy.",
            "Eating too much sugar can rise the risk of diabetes.",
            "Online courses help learners save their time and fee.",
            "He has a high chance to win the job interview.",
            "Living in big cities is very convenient to find jobs.",
            "Lỗi dùng từ không tự nhiên do dịch từng từ tiếng Việt sang tiếng Anh.",
            "Sai collocation phổ biến: dùng do mistakes thay vì make mistakes.",
            "Thêm s vào danh từ không đếm được như informations, advices, equipments.",
            "Lặp từ vựng đơn giản quá nhiều như good, bad, big, very, a lot of.",
            "Dùng từ sai văn phong học thuật (informal slang) trong bài Writing Task 2.",
            "The government should push the development of green energy.",
            "We must protect the nature and stop destroying wild animals.",
            "Reading books can open our mind and give us many experiences.",
            "He said me that the IELTS exam was very hard.",
            "This machine helps workers reduce their hard working.",
        ],
        "sentence_structure": [
            "Although the tuition fee is expensive, but many students still study abroad.",
            "Because air pollution is getting worse, so people wear masks.",
            "Many young people use smartphones, they spend hours on social media.",
            "Studying abroad is beneficial, however it is very costly for families.",
            "For example, cars, motorbikes, and buses in big cities.",
            "Which causes serious damage to the local ecosystem.",
            "When children playing games too much, losing focus in class.",
            "There are many people want to move to urban areas for better jobs.",
            "I think that people should use public transport it is cheaper.",
            "The reason why people commute is because housing is expensive.",
            "Not only students benefit from online learning, teachers also save time.",
            "While the percentage of coal fell, whereas solar energy increased.",
            "Having a university degree that helps graduates find good jobs.",
            "People who living in the countryside enjoy cleaner air.",
            "Despite he worked very hard, he did not pass the exam.",
            "In addition to reduce traffic congestion, public buses lower emissions.",
            "It is undeniable that technology which has transformed education.",
            "My hometown has many parks, people go there to exercise, it is peaceful.",
            "Because of the economy is growing, living standards are higher.",
            "Even though the weather was bad, yet the event continued.",
            "Lỗi câu run-on và comma splice nối hai mệnh đề độc lập bằng dấu phẩy.",
            "Lỗi dùng kép liên từ Although với But hoặc Because với So trong cùng một câu.",
            "Câu cụt thiếu động từ chính (sentence fragment), chỉ có cụm danh từ.",
            "Sai mệnh đề quan hệ: dùng there are many people want thay vì who want.",
            "Lỗi cấu trúc song song (parallelism) và sai mệnh đề nhượng bộ Despite + clause.",
            "Online shopping is convenient, therefore many shops are closing down.",
            "To improve English speaking skills, practicing every day.",
            "The movie which I watched it yesterday was really inspiring.",
            "Studying at home is flexible, on the other hand, students feel isolated.",
            "Although living in cities is stressful, however salaries are higher.",
        ],
        "coherence_cohesion": [
            "Firstly, cars cause pollution. Moreover, education is important. Therefore, schools need computers.",
            "In the other hand, working from home has several drawbacks for teams.",
            "Besides that, technology is good. Furthermore, it is fast. In addition, it is cheap.",
            "People should protect wildlife. It is very important for them because they need it.",
            "To begin with, I will talk about sports. Secondly, healthy food. Finally, sleep.",
            "However, smartphones are useful. However, they distract students. However, parents worry.",
            "In conclusion, there are both advantages and disadvantages, and I agree with this.",
            "The paragraph has no clear topic sentence and jumps between unrelated ideas.",
            "On the contrary, online learning saves money while traditional classes are face to face.",
            "As a result, students get tired. For instance, many people like traveling abroad.",
            "Firstly, the graph shows water use. Secondly, agriculture is big. Thirdly, industry is small.",
            "Using too many linking words mechanically at the start of every single sentence.",
            "The pronoun 'they' in the third sentence does not clearly refer to any noun.",
            "Ideas in body paragraph 2 repeat body paragraph 1 without logical progression.",
            "Missing transition between the cause of traffic jams and the proposed government solutions.",
            "In my essay I will discuss both views. First view is good. Second view is also good.",
            "At the same time, crime rates increased. By the way, tourists love local food.",
            "Moreover, pollution harms health. In contrast, hospitals are expensive.",
            "Đoạn thân bài thiếu câu chủ đề (topic sentence) khiến người đọc khó nắm ý chính.",
            "Lạm dụng từ nối máy móc Firstly, Secondly, Moreover ở đầu mọi câu.",
            "Sai cụm từ nối cố định: viết In the other hand thay vì On the other hand.",
            "Ý tưởng trong đoạn văn nhảy cóc, thiếu sự liên kết logic giữa luận điểm và ví dụ.",
            "Đại từ thay thế it, they, this không rõ ràng dẫn đến câu văn mơ hồ.",
            "Chuyển đoạn đột ngột, không có câu dẫn dắt giữa hai luận điểm đối lập.",
            "Mở bài và kết bài mâu thuẫn nhau về quan điểm đã trình bày.",
            "Last but not least, every coin has two sides so we should balance it.",
            "Basically, as you can see, thing is that internet is everywhere nowadays.",
            "This paragraph mixes advantages and solutions together without clear organization.",
            "Repetitive use of 'also' and 'and' to connect unrelated arguments.",
            "Weak referencing and lack of cohesive devices to link supporting explanations.",
        ],
        "task_response": [
            "The essay only wrote 180 words for Writing Task 2 which requires at least 250 words.",
            "Prompt asked to discuss both views and give your opinion, but the essay gave no opinion.",
            "Essay went off-topic by talking about general health instead of fast food advertising.",
            "Task 1 report listed every single number mechanically without an overview paragraph.",
            "Arguments are too general and lack specific explanations or relevant examples.",
            "Only answered the causes part of the two-part question and forgot to mention solutions.",
            "Thesis statement is unclear and contradicts the conclusion paragraph.",
            "Included personal opinions and assumptions in Academic Writing Task 1 chart description.",
            "Used memorized cliché sentences that do not directly answer the essay question.",
            "Failed to compare key trends and highest/lowest figures in the bar chart overview.",
            "Writing Task 1 was under 150 words and missed two categories from the table.",
            "The essay agrees and disagrees vaguely without taking a clear position.",
            "Example about my uncle's friend is too informal and does not support the main claim.",
            "Prompt asked whether benefits outweigh drawbacks, but essay only listed benefits.",
            "Superficial bullet-like ideas without explaining why or how the effect happens.",
            "Lạc đề: đề bài hỏi về thuế đường nhưng bài viết lại bàn chung chung về tập thể dục.",
            "Thiếu đoạn Overview tổng quan trong IELTS Writing Task 1.",
            "Chưa trả lời đủ hai câu hỏi của dạng Two-part question, bỏ sót phần giải pháp.",
            "Bài viết không nêu rõ quan điểm cá nhân dù đề yêu cầu Give your own opinion.",
            "Thiếu số từ quy định (dưới 250 từ Task 2 hoặc dưới 150 từ Task 1) nên bị trừ điểm nặng.",
            "Lập luận hời hợt, chỉ liệt kê ý tưởng mà không phân tích nguyên nhân hệ quả.",
            "Đưa ý kiến chủ quan vào bài miêu tả biểu đồ Writing Task 1.",
            "Copied the prompt word-for-word in the introduction without paraphrasing.",
            "Overgeneralized claim saying all teenagers are addicted to phones without nuance.",
            "Did not address the 'to what extent' part of the essay prompt.",
            "Task 1 map description ignored major changes in the northern area.",
            "Process diagram skipped two middle stages of recycling plastic bottles.",
            "Unsupported sweeping statements about government failure with zero reasoning.",
            "Essay answered a memorized topic about technology instead of the actual question on remote work.",
            "No clear stance throughout the essay despite choosing an agree/disagree prompt.",
        ],
        "spelling_wordform": [
            "The goverment should protect the enviroment for future generations.",
            "This new policy is very benefit for local economy growth.",
            "Students need to improve their confident when speaking English.",
            "Many people recieve false information from social media platforms.",
            "Online education is becoming increasingly popularity around the world.",
            "She worked very hardly to achieve a high IELTS score.",
            "The economic of the country depends heavily on tourism.",
            "Children should be educate about nutrition from an early age.",
            "There is a big different between urban and rural lifestyles.",
            "Air polution causes serious respiratory deseases in cities.",
            "He gave a very successfully presentation at the conference.",
            "Accomodation in big cities is becoming unaffordable for young workers.",
            "Technology has made communication more easier and conveniant.",
            "Parents play a crucial responsible in raising their kids.",
            "The develop of artificial intelligence brings new oppurtunities.",
            "Students often feel stressful and anxiety before final exams.",
            "Public transport is more safety and reliability than motorbikes.",
            "It is neccessary to seperate household waste before recycling.",
            "The companys profit increased dramatic in the last quarter.",
            "He is a very intelligence student with strong creative.",
            "Lỗi viết sai chính tả từ học thuật phổ biến: goverment, enviroment, accomodation.",
            "Nhầm lẫn từ loại danh từ, tính từ, trạng từ (benefit vs beneficial, different vs difference).",
            "Dùng sai tính từ đuôi -ed và -ing (interesting vs interested, boring vs bored).",
            "Sai so sánh hơn kép more easier, more better hoặc sai dạng từ loại sau động từ to be.",
            "Viết sai chính tả trong bài Listening điền từ khiến mất điểm đáng tiếc.",
            "Definetly we must take imediate action to reduce global warming.",
            "The society is facing a danger situation due to unemployment.",
            "Many young graduates lack of finance independence from parents.",
            "Tourism brings significant culture exchanges between nations.",
            "Her explaination was clear and easy to understand.",
        ],
        "listening_reading_trap": [
            "Wrote three words when the Listening instruction clearly stated NO MORE THAN TWO WORDS.",
            "Missed the plural 's' at the end of 'tickets' in Listening Section 1 form completion.",
            "Fell for speaker self-correction: chose Tuesday instead of Thursday after 'actually, sorry'.",
            "Confused FALSE with NOT GIVEN in Reading True/False/Not Given passage.",
            "Matched a Reading heading based on one identical keyword instead of the paragraph main idea.",
            "Missed the paraphrase: question used 'annual cost' while passage said 'yearly fee'.",
            "Wrote the currency symbol wrong or forgot 'million' unit in Listening table completion.",
            "Stopped listening too early in Section 3 multiple choice before the students rejected option A.",
            "Misspelled a dictated postcode and surname in Listening Part 1.",
            "Over-inferred outside knowledge in Reading instead of relying strictly on the passage text.",
            "Chose distractor number 15 instead of 50 in Listening Section 1 phone and price question.",
            "Failed to notice negative qualifier 'rarely' or 'only' in Reading statement.",
            "Selected a heading that only described a minor example rather than the whole paragraph.",
            "Forgot to transfer answers cleanly or shifted question numbers on the Reading answer sheet.",
            "Misunderstood map directions 'opposite', 'adjacent to', and 'just past the bridge' in Listening.",
            "Bị dính bẫy distractor trong IELTS Listening khi người nói tự sửa lại thông tin ngay sau đó.",
            "Nhầm lẫn giữa FALSE và NOT GIVEN trong bài Reading do tự suy diễn kiến thức ngoài.",
            "Viết quá giới hạn từ NO MORE THAN TWO WORDS AND/OR A NUMBER trong Listening và Reading.",
            "Bỏ sót âm cuối số nhiều s/es hoặc đơn vị đo lường khi điền từ bài nghe.",
            "Không nhận ra từ đồng nghĩa paraphrase giữa câu hỏi và đoạn văn Reading.",
            "Chọn đáp án theo từ khóa giống hệt (keyword matching trap) mà không đọc kỹ ngữ cảnh.",
            "Bị rối phương hướng bản đồ Listening Map Labelling (north-east, opposite, next to).",
            "Mất dấu vị trí trong Listening Section 4 lecture vì bỏ qua signpost words.",
            "Spent 30 minutes on Reading Passage 1 and ran out of time for Passage 3.",
            "Confused 'a few' with 'few' and 'most' with 'all' in Reading True False Not Given.",
            "Did not check grammar fit when filling blanks in Reading Summary Completion.",
            "Missed synonym mapping between 'elderly residents' and 'senior citizens' in Matching Info.",
            "Picked the first option mentioned in Listening MCQ without waiting for 'but' or 'however'.",
            "Copied a word that already appeared after the blank, creating duplicate words.",
            "Misread 'YES/NO/NOT GIVEN' instructions and wrote 'TRUE/FALSE' instead.",
        ],
    }


def generate_error_dataset(seed: int = 42) -> List[Dict[str, str]]:
    """Build an augmented labeled dataset for IELTS common error classification."""
    rng = np.random.default_rng(seed)
    base = _base_error_examples()
    rows: List[Dict[str, str]] = []

    prefixes_en = [
        "",
        "Student wrote: ",
        "In essay: ",
        "IELTS Writing feedback - ",
        "Speaking transcript: ",
        "Mock test mistake: ",
    ]
    prefixes_vi = [
        "",
        "Học sinh mắc lỗi: ",
        "Nhận xét giáo viên: ",
        "Bài làm bị trừ điểm vì: ",
    ]

    idx = 1
    for category, examples in base.items():
        for text in examples:
            rows.append({"error_id": f"ERR-{idx:04d}", "text": text, "category": category})
            idx += 1

            is_vi = any(
                ch in text
                for ch in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
            )
            prefix = str(rng.choice(prefixes_vi if is_vi else prefixes_en))
            if prefix:
                rows.append(
                    {
                        "error_id": f"ERR-{idx:04d}",
                        "text": f"{prefix}{text}",
                        "category": category,
                    }
                )
                idx += 1

    rng.shuffle(rows)
    return rows


def save_datasets(data_dir: Path, n_students: int = 1800, seed: int = 42) -> Tuple[Path, Path]:
    """Generate and persist both CSV datasets to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    students_path = data_dir / "ielts_students.csv"
    errors_path = data_dir / "ielts_errors.csv"

    student_rows = generate_student_dataset(n_samples=n_students, seed=seed)
    with students_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STUDENT_COLUMNS)
        writer.writeheader()
        writer.writerows(student_rows)

    error_rows = generate_error_dataset(seed=seed)
    with errors_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["error_id", "text", "category"])
        writer.writeheader()
        writer.writerows(error_rows)

    return students_path, errors_path
