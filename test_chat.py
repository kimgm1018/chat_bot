import requests

url = ''


# 처음에만 ID 입력
USER_ID = input("enter your id : ")

while True:
    user_input = input("YOU : ")
    if user_input == "0":
        break

    # 이후에는 자동으로 user_id 포함
    payload = {
        "user_id": USER_ID, 
        "message": user_input
    }

    res = requests.post(url, json=payload)
    print(res.json().get("answer", ''))
