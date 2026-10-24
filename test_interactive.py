# ==============================================================================
# Interactive Terminal Email Tester (Type your own emails)
# ==============================================================================

from simple_spam_detector import model, vectorizer

print("=" * 55)
print("🤖 Spam Email Detector - Interactive Terminal Mode")
print("Type or paste any email message to test (Press Ctrl+C to quit)")
print("=" * 55 + "\n")

while True:
    try:
        user_text = input("Enter email text: ").strip()
        if not user_text:
            continue
        
        features = vectorizer.transform([user_text])
        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        
        spam_pct = probs[1] * 100
        if pred == 1:
            print(f"-> 🚨 SPAM DETECTED ({spam_pct:.1f}% Spam Probability)\n")
        else:
            print(f"-> ✅ SAFE EMAIL ({100 - spam_pct:.1f}% Safe Probability)\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting. Goodbye!")
        break
