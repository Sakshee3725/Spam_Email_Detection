# ==============================================================================
# Spam Email Detection Model (Scikit-Learn Multinomial Naive Bayes)
# ==============================================================================

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Training Dataset (Sample Emails from Real Data)
# 1 = SPAM, 0 = LEGITIMATE (HAM)
training_emails = [
    # --- Spam Emails ---
    'URGENT: Your PayPal account has been suspended! Verify password now',
    'CONFIDENTIAL: You have inherited $18.5 Million USD wire transfer',
    'INVOICE: $499.00 deducted from your bank account call immediately to refund',
    'Earn $3,500 daily guaranteed automated bitcoin crypto mining robot',
    'Urgent wire request settle closing deposit transfer $48,000 today',
    'Congratulations you won 2.5 million euros in the lottery claim prize',
    'Action required: Your Office 365 password expires today review credentials',
    'Pre-approved: $25,000 unsecured cash loan deposited today zero credit check',
    'Undelivered parcel unpaid customs fee click here to pay online',
    'Security alert: Your Apple ID locked unauthorized login attempt in China',
    'You have been chosen: Claim your free brand new iPhone 16 Pro today',
    'Unauthorized access attempt on your checking account verify card PIN',
    '90% OFF clearance sale limited time flash offer buy now',

    # --- Legitimate (Ham) Emails ---
    'Pull request #142: Fix connection pooling timeout on database',
    'Your Amazon.com order #114-8921 has been placed and confirmed',
    'Sprint planning agenda for Thursday 10:00 AM in conference room',
    'United Airlines eTicket flight confirmation San Francisco to Chicago',
    'Hey are you free for dinner this Saturday evening let us catch up',
    'Your monthly processing fee invoice and statement is ready for download',
    'Google Calendar invitation: Machine learning architecture sync Friday 3pm',
    'Morning news rundown: tech markets rally and quarterly earnings report',
    'How engineering teams scale real-time message database architecture',
    'Guest lecture: Natural language processing advances by Prof. Manning',
    'Quick question: did you finish question 4 for homework assignment',
    'Hello! Just checking in to see how you are doing have a great week',
    'Doctor appointment reminder for Friday at 2:00 PM please arrive early'
]

# Add varied examples of everyday conversations and common scam messages.
# A small, balanced dataset can otherwise overreact to individual words.
training_emails.extend([
    # Additional spam examples
    'Claim your free $500 gift card now',
    'You are a winner claim your free gift card today',
    'Act now to claim a cash prize click the link',
    'Congratulations you have won a free reward claim now',
    'Limited time offer earn money fast guaranteed',
    'Verify your account immediately to avoid suspension',
    'You have been selected for a free prize claim today',
    'Exclusive offer buy now and receive a bonus',
    # Additional safe everyday messages
    'Hey are we meeting for coffee today?',
    'Are we still meeting today for coffee?',
    'Hi, would you like to grab coffee this afternoon?',
    'Hey, what time should we meet today?',
    'Good morning, are you free to meet later?',
    'Can we meet for lunch today?',
    'Hey, how are you doing today?',
    'Are you coming to class today?',
    'Let me know when you reach the cafe',
    'Would you like to catch up over coffee this week?',
    'Hi, just checking whether our meeting is still on',
    'I will meet you at the coffee shop at five',
    'Thanks for your message, see you later',
    'Can we discuss the assignment after class?',
    'Are you available for a quick call today?',
    'Hey, I am running late. Please wait for me',
    'Please send me the notes from today class',
    'See you tomorrow morning',
    'Would you like to join us for dinner?',
    'The meeting has been moved to Friday afternoon',
    'I have reached the cafe. Where are you?',
])

# 1 = Spam, 0 = Safe
labels = [1] * 13 + [0] * 13 + [1] * 8 + [0] * 21

# 2. Text Vectorization (Count word frequencies)
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(training_emails)

# 3. Train Multinomial Naive Bayes Classifier
model = MultinomialNB(alpha=1.0)
model.fit(X_train, labels)

print("=" * 60)
print(f"✅ Model successfully trained on {len(training_emails)} sample emails!")
print("=" * 60 + "\n")

# 4. Function to classify any new email
def check_email(text):
    features = vectorizer.transform([text])
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    is_spam = pred == 1
    spam_pct = probs[1] * 100
    
    print("-" * 60)
    print("Email Text:", text)
    if is_spam:
        print(f"🚨 VERDICT: SPAM DETECTED! ({spam_pct:.1f}% Spam Probability)")
    else:
        print(f"✅ VERDICT: SAFE EMAIL! ({100 - spam_pct:.1f}% Ham Probability)")
    print("-" * 60 + "\n")

# 5. Run tests on sample emails
if __name__ == '__main__':
    print("Running Sample Tests from Dataset:\n")

    check_email("Urgent: Your account is suspended, click here to verify password immediately.")
    check_email("Hi team, attached is the report for our Thursday planning meeting, thanks.")
    check_email("Congratulations winner! Claim your free cash reward and lottery prize now.")
    check_email("Hey did you finish the math homework assignment for class tomorrow?")
