# NTUH Portal Login

`portal_log_in.py` 以 Selenium 登入 NTUH Portal，並內嵌六碼 CAPTCHA OCR 模型。不需安裝 Tesseract，也不需另外下載 OCR 模型檔。

## 環境

- Python 3.10 以上
- Google Chrome
- 可連線 NTUH Portal 的網路環境

## 安裝

建議直接從 GitHub 安裝，這會同時安裝必要套件：

```powershell
python -m pip install git+https://github.com/ntuhdermatology-collab/NTUH_portal_log_in.git
```

專案開發者也可在 clone 後使用 `python -m pip install -r requirements.txt`。

## 使用

```python
from getpass import getpass

from portal_log_in import log_in


person = input("User ID: ")
password = getpass("Password: ")

driver = log_in(person, password, system=0, show=1, headless=1)
try:
    # 在這裡操作已登入的 Selenium driver
    pass
finally:
    driver.quit()
```

`system` 可使用以下數值：

- `0`：一般
- `1`：門診
- `2`：住院
- `3`：急診

`show=1` 會顯示 OCR 與登入狀態；`headless=1` 會以無頭模式啟動 Chrome。

## 只下載單一程式檔

若環境已安裝必要套件，可用 Python 標準庫下載最新的 `portal_log_in.py`，不需要另外安裝 `requests`：

```python
from urllib.request import urlretrieve


url = "https://raw.githubusercontent.com/ntuhdermatology-collab/NTUH_portal_log_in/main/portal_log_in.py"
urlretrieve(url, "portal_log_in.py")

from portal_log_in import log_in
```

此方式只會下載程式檔，不會安裝 `selenium`、`numpy` 與 `opencv-python`。

## 安全

請勿將帳號、密碼、`config.json` 或院內資料上傳到 GitHub。本專案的 `.gitignore` 已預設排除本機設定、暫存驗證碼與本機 Excel 資料。僅應在獲得授權的環境中使用。
