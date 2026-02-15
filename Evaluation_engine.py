import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# 1️⃣ Load Models
# -----------------------------

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# -----------------------------
# 2️⃣ Text Preprocessing
# -----------------------------

def preprocess(text):
    doc = nlp(text)
    cleaned_tokens = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            cleaned_tokens.append(token.lemma_.lower())

    return " ".join(cleaned_tokens)


# -----------------------------
# 3️⃣ Concept Extraction
# -----------------------------

def extract_concepts(text):
    doc = nlp(text)
    concepts = []

    for chunk in doc.noun_chunks:
        concepts.append(chunk.lemma_.lower())

    return list(set(concepts))



# -----------------------------
# 4️⃣ Semantic Similarity
# -----------------------------

def calculate_similarity(ans1, ans2):
    emb1 = embedding_model.encode([ans1])
    emb2 = embedding_model.encode([ans2])

    score = cosine_similarity(emb1, emb2)[0][0]
    return round(score * 100, 2)


# -----------------------------
# 5️⃣ Concept Coverage
# -----------------------------

def concept_coverage(model_ans, student_ans):
    model_concepts = extract_concepts(model_ans)
    student_concepts = extract_concepts(student_ans)

    matched = set(model_concepts).intersection(set(student_concepts))

    if len(model_concepts) == 0:
        return 0, model_concepts, student_concepts, matched

    coverage = len(matched) / len(model_concepts)
    return round(coverage * 100, 2), model_concepts, student_concepts, matched


# -----------------------------
# 6️⃣ Viva Question Generation
# -----------------------------

def generate_viva_questions(missing_concepts, difficulty="basic"):
    questions = []

    for concept in missing_concepts:
        if difficulty == "basic":
            questions.append(f"What is {concept}?")

        elif difficulty == "intermediate":
            questions.append(f"How does {concept} affect database design?")

        elif difficulty == "advanced":
            questions.append(f"Explain the impact of {concept} in real-world database systems.")

    return questions[:3]



# -----------------------------
# 7️⃣ Final Score Calculation
# -----------------------------

def final_score(similarity, coverage):
    if similarity > 80:
        weight_sim = 0.7
    else:
        weight_sim = 0.6

    weight_cov = 1 - weight_sim

    score = weight_sim * similarity + weight_cov * coverage
    return round(score, 2)


def confidence_level(score):
    if score > 80:
        return "High Understanding"
    elif score > 50:
        return "Moderate Understanding"
    else:
        return "Low Understanding"




# -----------------------------
# 🔥 MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":

    model_answer = "Normalization reduces redundancy in relational databases."
    student_answer = "Normalization helps remove repeated data from database tables."

    # Preprocess
    clean_model = preprocess(model_answer)
    clean_student = preprocess(student_answer)

    print("\nCleaned Model Answer:", clean_model)
    print("Cleaned Student Answer:", clean_student)

    # Similarity
    similarity_score = calculate_similarity(clean_model, clean_student)
    print("\nSemantic Similarity Score:", similarity_score)

    # Concept Coverage
    coverage_score, model_concepts, student_concepts, matched = concept_coverage(model_answer, student_answer)

    print("\nModel Concepts:", model_concepts)
    print("Student Concepts:", student_concepts)
    print("Matched Concepts:", matched)
    print("Concept Coverage Score:", coverage_score)


    # Missing concepts
    missing = set(model_concepts) - set(student_concepts)

    # Viva Questions
    viva_questions = generate_viva_questions(missing, difficulty="intermediate")

    print("\nGenerated Viva Questions:")
    for q in viva_questions:
        print("-", q)

    # Final Score
    final = final_score(similarity_score, coverage_score)
    print("\nFinal Evaluation Score:", final)
    
    
    # Confidence Level
    confidence = confidence_level(final)
    print("Confidence Level:", confidence)

