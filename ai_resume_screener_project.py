from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("---AI RESUME SCREENER---")

with open("resume.txt","r",encoding="utf-8") as f:
    resume=f.read()

with open("job.txt","r",encoding="utf-8") as f:
    job_description=f.read()

vectorizer = TfidfVectorizer(stop_words="english",ngram_range=(1,2))
vectors = vectorizer.fit_transform([resume, job_description])

similarity = cosine_similarity(vectors[0:1], vectors[1:2])

stop_words=text.ENGLISH_STOP_WORDS

print("\nMatch Score:", round(similarity[0][0] * 100, 2), "%")

resume_words=set(resume.lower().split())-stop_words
job_words=set(job_description.lower().split())-stop_words

missing=job_words-resume_words

print("\nMissing Keywords(important ones):")
for word in list(missing)[:10]:
    print("-",word)
