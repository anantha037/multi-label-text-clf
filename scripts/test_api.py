import requests
import sys

BASE_URL = "http://localhost:8000"

def test_api():
    print("====================================")
    print("Testing GET /health")
    try:
        health_res = requests.get(f"{BASE_URL}/health")
        print("Status Code:", health_res.status_code)
        print("Response:", health_res.json())
    except Exception as e:
        print("Error connecting to server:", e)
        sys.exit(1)
    print("====================================\n")
    
    print("====================================")
    print("Testing GET /labels")
    labels_res = requests.get(f"{BASE_URL}/labels")
    print("Status Code:", labels_res.status_code)
    try:
        labels = labels_res.json()
        print(f"Number of Labels: {len(labels)}")
        print(f"Labels: {labels}")
    except Exception as e:
        print("Response:", labels_res.text)
    print("====================================\n")
    
    print("====================================")
    print("Testing POST /predict")
    sample_text = "I am so grateful and happy today"
    print(f"Input text: '{sample_text}'")
    predict_res = requests.post(f"{BASE_URL}/predict", json={"text": sample_text})
    print("Status Code:", predict_res.status_code)
    try:
        data = predict_res.json()
        print(f"Predicted Labels: {data.get('labels')}")
        
        scores = data.get("scores", {})
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 Scores:")
        for label, score in top_scores:
            print(f"  {label}: {score:.4f}")
    except Exception as e:
        print("Response:", predict_res.text)
    print("====================================\n")

if __name__ == "__main__":
    test_api()
