"""
Quick test for Resend API
"""
import resend
import os
from dotenv import load_dotenv

load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")

print("Testing Resend API Key...")
print("="*50)

try:
    params = {
        "from": "onboarding@resend.dev",
        "to": ["delivered@resend.dev"],  # Resend's test address
        "subject": "Test Email from Deep Research Agent",
        "html": "<h1>Hello!</h1><p>This is a test email. Your Resend API is working! Hooray.....🎉</p>"
    }
    
    email = resend.Emails.send(params)
    print("✅ SUCCESS! Email sent!")
    print(f"📧 Email ID: {email['id']}")
    print("\nCheck your Resend dashboard at https://resend.com/emails")
    print("You should see this email in the 'Emails' section!")
    
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    print("\nPossible issues:")
    print("1. API key is incorrect")
    print("2. API key doesn't have permission to send emails")
    print("3. Check your .env file")

print("="*50)
