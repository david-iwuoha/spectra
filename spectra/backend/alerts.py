import resend
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")

ALERT_RECIPIENTS = [
    "davidiw032@gmail.com",  # replace with your real email
]

def send_spill_alert(detection: dict):
    """Send email alert when a high-confidence spill is detected."""

    confidence = detection.get("confidence", 0)
    area_km2 = detection.get("area_km2", 0)
    detected_at = detection.get("detected_at", "")
    scan_id = detection.get("id", "")
    scene = detection.get("scene", "")

    # Format time nicely
    try:
        dt = datetime.fromisoformat(detected_at)
        time_str = dt.strftime("%B %d, %Y at %H:%M UTC")
    except:
        time_str = detected_at

    subject = f"🚨 Spectra Alert — Oil Spill Detected ({confidence}% confidence)"

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: #0F6E56; padding: 24px; border-radius: 8px 8px 0 0;">
            <h1 style="color: white; margin: 0; font-size: 24px;">SPECTRA</h1>
            <p style="color: #9FE1CB; margin: 4px 0 0;">AI-Powered Oil Spill Detection</p>
        </div>

        <div style="background: #FFF3CD; padding: 16px 24px; border-left: 4px solid #FF6B35;">
            <h2 style="color: #FF6B35; margin: 0;">⚠️ Oil Spill Detected</h2>
        </div>

        <div style="background: #f9f9f9; padding: 24px;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #666; font-weight: bold;">Detection ID</td>
                    <td style="padding: 12px 0; color: #333;">{scan_id}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #666; font-weight: bold;">Detected At</td>
                    <td style="padding: 12px 0; color: #333;">{time_str}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #666; font-weight: bold;">Confidence</td>
                    <td style="padding: 12px 0;">
                        <span style="background: #FF6B35; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;">
                            {confidence}%
                        </span>
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #666; font-weight: bold;">Estimated Area</td>
                    <td style="padding: 12px 0; color: #333;">{area_km2} km²</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #666; font-weight: bold;">Scene</td>
                    <td style="padding: 12px 0; color: #333;">{scene}</td>
                </tr>
                <tr>
                    <td style="padding: 12px 0; color: #666; font-weight: bold;">Region</td>
                    <td style="padding: 12px 0; color: #333;">Niger Delta, Nigeria</td>
                </tr>
            </table>
        </div>

        <div style="background: #E1F5EE; padding: 16px 24px;">
            <p style="margin: 0; color: #0F6E56; font-size: 14px;">
                This alert was generated automatically by Spectra's AI detection engine.
                No human intervention required. Evidence archived with timestamp and coordinates.
            </p>
        </div>

        <div style="background: #0F6E56; padding: 16px 24px; border-radius: 0 0 8px 8px; text-align: center;">
            <p style="color: #9FE1CB; margin: 0; font-size: 12px;">
                Spectra — A whistleblower that runs on physics.
            </p>
        </div>
    </div>
    """

    try:
        response = resend.Emails.send({
            "from": "Spectra Alerts <onboarding@resend.dev>",
            "to": ALERT_RECIPIENTS,
            "subject": subject,
            "html": html_body,
        })
        print(f"Alert sent successfully. ID: {response['id']}")
        return True
    except Exception as e:
        print(f"Alert failed: {e}")
        return False


if __name__ == "__main__":
    # Test alert
    test_detection = {
        "id": "test-001",
        "scene": "S1A_IW_GRDH_Niger_Delta_20240117",
        "detected_at": datetime.utcnow().isoformat(),
        "confidence": 68.14,
        "area_km2": 0.4653,
        "spill_pixels": 4653,
    }
    print("Sending test alert...")
    send_spill_alert(test_detection)
