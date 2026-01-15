"""
Twilio SMS Notification System for AquaGuard Pro
Sends emergency alerts when water quality is detected as hazardous
"""

import os
import json
from datetime import datetime
from twilio.rest import Client
import streamlit as st

# Twilio Configuration
TWILIO_ACCOUNT_SID = "AC462224da3e6f655251f6bf2cd7631a65"
TWILIO_AUTH_TOKEN = "b364e1a75fc34c1eb275b62c8754c68a"
TWILIO_PHONE_NUMBER = "+16282448340"

class TwilioNotificationSystem:
    def __init__(self):
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.notification_history_file = "notification_history.json"
        
    def load_notification_history(self):
        """Load previous notification history to avoid spam"""
        try:
            if os.path.exists(self.notification_history_file):
                with open(self.notification_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading notification history: {e}")
            return []
    
    def save_notification_history(self, history):
        """Save notification history"""
        try:
            with open(self.notification_history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Error saving notification history: {e}")
    
    def should_send_notification(self, prediction_class, sensor_data):
        """Determine if notification should be sent (only for hazardous conditions)"""
        if prediction_class != "Hazardous":
            return False, "Not a hazardous condition"
        
        history = self.load_notification_history()
        now = datetime.now()
        
        # Check if we sent a notification in the last 30 minutes for similar conditions
        for record in history:
            record_time = datetime.fromisoformat(record['timestamp'])
            time_diff = (now - record_time).total_seconds() / 60  # minutes
            
            if time_diff < 30:  # 30-minute cooldown
                return False, f"Recent notification sent {int(time_diff)} minutes ago"
        
        return True, "Hazardous condition detected"
    
    def format_hazardous_message(self, sensor_data, confidence):
        """Format the hazardous water alert message"""
        # Get concerning parameters
        critical_params = []
        
        # Analyze each parameter
        if sensor_data['PH'].iloc[0] < 6.0 or sensor_data['PH'].iloc[0] > 9.0:
            critical_params.append(f"pH: {sensor_data['PH'].iloc[0]:.2f}")
        
        if sensor_data['D.O. (mg/l)'].iloc[0] < 3.0:
            critical_params.append(f"Dissolved O₂: {sensor_data['D.O. (mg/l)'].iloc[0]:.1f} mg/L")
        
        if sensor_data['B.O.D. (mg/l)'].iloc[0] > 5.0:
            critical_params.append(f"BOD: {sensor_data['B.O.D. (mg/l)'].iloc[0]:.1f} mg/L")
        
        if sensor_data['FECAL COLIFORM (MPN/100ml)'].iloc[0] > 100:
            critical_params.append(f"Fecal Coliform: {sensor_data['FECAL COLIFORM (MPN/100ml)'].iloc[0]:.0f} MPN/100ml")
        
        critical_text = ", ".join(critical_params) if critical_params else "Multiple critical parameters"
        
        message = f"""🚨 WATER SAFETY ALERT

HAZARDOUS WATER DETECTED
Immediate action required!

Critical Parameters:
{critical_text}

Confidence: {confidence:.1%}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AquaGuard Pro Monitoring System
Reply STOP to opt out"""
        
        return message
    
    def send_notification(self, to_phone, sensor_data, confidence):
        """Send SMS notification for hazardous water"""
        try:
            message_body = self.format_hazardous_message(sensor_data, confidence)
            
            # Send the message
            message = self.client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to="+601127267920"
            )
            
            # Save to history
            history = self.load_notification_history()
            history.append({
                'timestamp': datetime.now().isoformat(),
                'to': to_phone,
                'message_sid': message.sid,
                'prediction': 'Hazardous',
                'confidence': confidence,
                'status': 'sent'
            })
            
            # Keep only last 100 records
            if len(history) > 100:
                history = history[-100:]
            
            self.save_notification_history(history)
            
            return True, f"Notification sent successfully! SID: {message.sid}"
            
        except Exception as e:
            # Save failed attempt to history
            history = self.load_notification_history()
            history.append({
                'timestamp': datetime.now().isoformat(),
                'to': to_phone,
                'prediction': 'Hazardous',
                'confidence': confidence,
                'status': 'failed',
                'error': str(e)
            })
            self.save_notification_history(history)
            
            return False, f"Failed to send notification: {e}"
    
    def load_notification_config(self):
        """Load notification configuration"""
        try:
            if os.path.exists("notification_config.json"):
                with open("notification_config.json", 'r') as f:
                    return json.load(f)
            return {"phone_numbers": []}
        except:
            return {"phone_numbers": []}
    
    def save_notification_config(self, config):
        """Save notification configuration"""
        try:
            with open("notification_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            st.error(f"Error saving configuration: {e}")
    
    def get_notification_stats(self):
        """Get notification statistics"""
        history = self.load_notification_history()
        
        total_sent = len([h for h in history if h.get('status') == 'sent'])
        total_failed = len([h for h in history if h.get('status') == 'failed'])
        
        # Recent activity (last 24 hours)
        now = datetime.now()
        recent_24h = [h for h in history 
                     if (now - datetime.fromisoformat(h['timestamp'])).total_seconds() < 86400]
        
        return {
            'total_sent': total_sent,
            'total_failed': total_failed,
            'recent_24h': len(recent_24h),
            'last_notification': history[-1]['timestamp'] if history else None
        }

# Global instance
notification_system = TwilioNotificationSystem()

def send_hazardous_water_alert(sensor_data, confidence, to_phone=None):
    """Main function to send hazardous water alert"""
    prediction = "Hazardous"  # This would come from your ML model
    
    # Check if notification should be sent
    should_send, reason = notification_system.should_send_notification(prediction, sensor_data)
    
    if not should_send:
        return False, reason
    
    # Use default phone if none provided
    if not to_phone:
        config = notification_system.load_notification_config()
        phone_numbers = config.get('phone_numbers', [])
        if not phone_numbers:
            return False, "No phone numbers configured for notifications"
        to_phone = phone_numbers[0]  # Use first configured number
    
    # Send the notification
    success, message = notification_system.send_notification(to_phone, sensor_data, confidence)
    return success, message

def configure_notification_numbers(phone_numbers):
    """Configure phone numbers for notifications"""
    config = notification_system.load_notification_config()
    config['phone_numbers'] = phone_numbers
    notification_system.save_notification_config(config)
    return True

def get_notification_status():
    """Get current notification system status"""
    stats = notification_system.get_notification_stats()
    config = notification_system.load_notification_config()
    
    return {
        'stats': stats,
        'configured_numbers': config.get('phone_numbers', []),
        'twilio_connected': True  # Assume connected if no errors
    }