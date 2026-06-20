import requests
import sys

TOKEN = "8391992002:AAGm3YU39dUz3dPUHCO601c_yh1RHZMFsho"
CHAT_ID = "345828071"

def test_telegram():
    print(f"[*] Testing Telegram Token: {TOKEN[:5]}...{TOKEN[-5:]}")
    print(f"[*] Target Chat ID: {CHAT_ID}")
    
    url = f"https://api.telegram.org/bot{TOKEN}/getMe"
    try:
        r = requests.get(url, timeout=10)
        print(f"[*] getMe Status: {r.status_code}")
        print(f"[*] getMe Response: {r.text}")
        
        if r.status_code != 200:
            print("[!] Token invalid or API error!")
            return

        url_msg = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": "🔔 Test Message from Debug Script"}
        r_msg = requests.post(url_msg, data=data, timeout=10)
        print(f"[*] sendMessage Status: {r_msg.status_code}")
        print(f"[*] sendMessage Response: {r_msg.text}")

    except Exception as e:
        print(f"[!] Connection Error: {e}")

if __name__ == "__main__":
    test_telegram()
