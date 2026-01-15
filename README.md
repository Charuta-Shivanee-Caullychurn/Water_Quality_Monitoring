# Water_Quality_Monitoring
This is a machine learning based water classification system which also extends to a simulation of a real-time system.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your Twilio credentials:
     - `TWILIO_ACCOUNT_SID`: Your Twilio Account SID
     - `TWILIO_AUTH_TOKEN`: Your Twilio Auth Token
     - `TWILIO_PHONE_NUMBER`: Your Twilio phone number
   - Optionally set `DEFAULT_RECIPIENT_PHONE` for default SMS recipient
4. Run the application
