import tkinter as tk
import re
from tkinter import ttk, messagebox, filedialog, simpledialog
import sqlite3
import math
import csv
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import threading
import json

# --- AI 引擎偵測區 ---
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np

    HAS_SKLEARN = True
    LOCAL_AI_STATUS = "✨ 本地進階模式 (Scikit-learn)"
except ImportError:
    HAS_SKLEARN = False
    LOCAL_AI_STATUS = "⚠️ 本地標準模式 (內建演算法)"


# --- 資料庫管理類別 ---
class DatabaseManager:
    def __init__(self, db_name="finance.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                category TEXT,
                item TEXT,
                amount INTEGER,
                type TEXT
            )
        """)
        self.conn.commit()

    def add_record(self, date, category, item, amount, r_type):
        self.cursor.execute("INSERT INTO records (date, category, item, amount, type) VALUES (?, ?, ?, ?, ?)",
                            (date, category, item, amount, r_type))
        self.conn.commit()

    def delete_record(self, record_id):
        self.cursor.execute("DELETE FROM records WHERE id=?", (record_id,))
        self.conn.commit()

    def fetch_all(self):
        self.cursor.execute("SELECT * FROM records ORDER BY date DESC")
        return self.cursor.fetchall()

    def get_summary(self):
        self.cursor.execute("SELECT category, SUM(amount) FROM records WHERE type='支出' GROUP BY category")
        return self.cursor.fetchall()

    def get_daily_expenses(self):
        self.cursor.execute("SELECT date, SUM(amount) FROM records WHERE type='支出' GROUP BY date ORDER BY date")
        return self.cursor.fetchall()

    def get_recent_records_text(self, limit=10):
        self.cursor.execute(
            "SELECT date, category, item, amount FROM records WHERE type='支出' ORDER BY date DESC LIMIT ?", (limit,))
        rows = self.cursor.fetchall()
        text_data = ""
        for r in rows:
            text_data += f"- {r[0]} [{r[1]}] {r[2]}: ${r[3]}\n"
        return text_data


# --- AI 與 爬蟲工具類別 ---
class SmartTools:
    @staticmethod
    def get_exchange_rates():
        """抓取臺灣銀行即時匯率：USD/JPY 現金賣出"""
        url = "https://rate.bot.com.tw/xrt?Lang=zh-TW"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        def to_float(s: str):
            s = s.replace(",", "").strip()
            if s in ("", "-"):
                return None
            m = re.search(r"-?\d+(\.\d+)?", s)
            return float(m.group(0)) if m else None

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            rates = {}
            for code in ("USD", "JPY"):
                # 找到包含 (USD)/(JPY) 的那格（比完全比對中文名稱穩很多）
                td = soup.find("td", string=re.compile(rf"\({code}\)"))
                if not td:
                    # fallback：有時候文字不在 td.string（被包在子標籤），就掃整列
                    for tr in soup.find_all("tr"):
                        if f"({code})" in tr.get_text(" ", strip=True):
                            td = tr.find("td")
                            break
                if not td:
                    raise RuntimeError(f"找不到 {code} 的列，可能網頁結構改了或被擋")

                tr = td.find_parent("tr")
                tds = [x.get_text(" ", strip=True) for x in tr.find_all("td")]

                # 通常格式：幣別 / 現金買入 / 現金賣出 / 即期買入 / 即期賣出 ...
                rate = to_float(tds[2]) if len(tds) >= 3 else None
                if rate is None:
                    # 再 fallback：抓到的數字裡面選「第二個數字」當現金賣出
                    nums = [to_float(x) for x in tds[1:]]
                    nums = [n for n in nums if n is not None]
                    rate = nums[1] if len(nums) >= 2 else None

                if rate is None:
                    raise RuntimeError(f"{code} 匯率解析失敗：{tds}")

                rates[code] = rate

            return rates

        except Exception as e:
            # 把真正錯誤帶回去，方便你定位（例如解析不到 / 403 / timeout）
            return {"Error": str(e)}

    @staticmethod
    def call_gemini_api(api_key, expense_text, total_expense, max_retries=3):
        model = "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        prompt = f"""
    你是一位專業的財務理財顧問。
    以下是使用者最近的消費紀錄 (總支出: ${total_expense}):
    {expense_text}
    請用繁體中文，針對這些消費習慣提供一段簡短、幽默且一針見血的理財建議 (100字以內)。
    """.strip()

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 200}  # 減少輸出 tokens
        }

        for attempt in range(max_retries + 1):
            r = requests.post(url, json=payload, timeout=20)

            if r.status_code == 200:
                data = r.json()
                cands = data.get("candidates", [])
                return cands[0]["content"]["parts"][0].get("text", "(無文字回覆)") if cands else "API 回覆無內容"

            # ✅ 429：依照 retryDelay 等一下再重試
            if r.status_code == 429:
                wait_s = None
                try:
                    data = r.json()
                    details = data.get("error", {}).get("details", [])
                    for d in details:
                        if d.get("@type", "").endswith("google.rpc.RetryInfo"):
                            m = re.search(r"(\d+)s", d.get("retryDelay", ""))
                            if m: wait_s = int(m.group(1))
                            break
                except:
                    pass

                if wait_s is None:
                    wait_s = min(2 ** attempt * 2, 20)  # fallback: 2,4,8... 最多 20 秒

                time.sleep(wait_s)
                continue

            # 其他錯誤：把內容吐出來方便你看
            return f"API 呼叫失敗 (HTTP {r.status_code})\n{r.text}"

        return "API 429：重試次數已用完（配額/速率限制仍未恢復）"

    @staticmethod
    def local_ai_prediction(data):
        if len(data) < 3:
            return "❌ 資料不足，無法進行趨勢分析。", 0
        dates = [datetime.strptime(d[0], "%Y-%m-%d").timestamp() for d in data]
        amounts = [d[1] for d in data]
        start_time = dates[0]
        x_days = [(d - start_time) / 86400 for d in dates]
        y_amount = amounts
        next_day_index = x_days[-1] + 1

        if HAS_SKLEARN:
            try:
                X = np.array(x_days).reshape(-1, 1)
                y = np.array(y_amount)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)
                next_X = poly.transform([[next_day_index]])
                prediction = model.predict(next_X)[0]
                return f"[本地分析]\n預測明日支出金額：${int(prediction)}", prediction
            except:
                pass

        n = len(x_days)
        sum_x = sum(x_days)
        sum_y = sum(y_amount)
        sum_xy = sum(xi * yi for xi, yi in zip(x_days, y_amount))
        sum_x2 = sum(xi ** 2 for xi in x_days)
        try:
            m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            b = (sum_y - m * sum_x) / n
            prediction = m * next_day_index + b
            return f"[本地分析]\n預測明日支出金額：${int(prediction)}", prediction
        except ZeroDivisionError:
            return "資料變異度不足", 0


# --- 主應用程式類別 ---
class SmartFinanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python AI 雲端財務管家 (API 版)")
        self.root.geometry("1050x700")
        self.root.resizable(False, False)

        self.db = DatabaseManager()
        self.api_key = ""
        self.current_rates = {}  # 儲存即時匯率

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Treeview.Heading", font=("微軟正黑體", 10, "bold"))

        self.create_widgets()
        self.refresh_data()

        threading.Thread(target=self.update_exchange_rates, daemon=True).start()

    def create_widgets(self):
        left_panel = tk.Frame(self.root, bg="#f8f9fa", padx=20, pady=20, relief="groove", bd=1)
        left_panel.place(x=0, y=0, width=350, height=700)

        tk.Label(left_panel, text="📝 記帳控制台", font=("微軟正黑體", 16, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
            anchor="w", pady=(0, 15))

        self.create_input_field(left_panel, "📅 日期:", datetime.now().strftime("%Y-%m-%d"), "date")
        self.create_input_field(left_panel, "💰 類型:", ["支出", "收入"], "type")
        self.create_input_field(left_panel, "🏷️ 分類:",
                                ["飲食", "交通", "娛樂", "購物", "居住", "醫療", "薪資", "其他"], "category")
        self.create_input_field(left_panel, "📝 項目:", "", "item")

        # --- 金額與幣別輸入區 ---
        tk.Label(left_panel, text="💲 金額 & 幣別:", bg="#f8f9fa", font=("微軟正黑體", 10)).pack(anchor="w", pady=(2, 0))
        amt_frame = tk.Frame(left_panel, bg="#f8f9fa")
        amt_frame.pack(fill="x", pady=2)

        self.entry_amount = ttk.Entry(amt_frame, width=15)
        self.entry_amount.pack(side="left", fill="x", expand=True)

        self.combo_currency = ttk.Combobox(amt_frame, values=["TWD", "USD", "JPY"], width=5, state="readonly")
        self.combo_currency.current(0)
        self.combo_currency.pack(side="left", padx=(5, 0))

        # 按鈕
        btn_frame = tk.Frame(left_panel, bg="#f8f9fa")
        btn_frame.pack(pady=15, fill="x")
        tk.Button(btn_frame, text="新增紀錄", bg="#27ae60", fg="white", font=("微軟正黑體", 10, "bold"),
                  command=self.add_record, relief="flat").pack(side="left", expand=True, fill="x", padx=2, ipady=5)
        tk.Button(btn_frame, text="刪除選取", bg="#e74c3c", fg="white", font=("微軟正黑體", 10, "bold"),
                  command=self.delete_record, relief="flat").pack(side="left", expand=True, fill="x", padx=2, ipady=5)
        ttk.Button(left_panel, text="匯出 CSV 報表", command=self.export_csv).pack(fill="x", pady=5)

        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=15)

        tk.Label(left_panel, text="☁️ 雲端 AI 財務長", font=("微軟正黑體", 14, "bold"), bg="#f8f9fa",
                 fg="#8e44ad").pack(anchor="w")
        self.btn_api = ttk.Button(left_panel, text="🔑 設定 Google Gemini API Key", command=self.set_api_key)
        self.btn_api.pack(fill="x", pady=5)
        self.lbl_api_status = tk.Label(left_panel, text="尚未設定 Key (使用本地模式)", font=("微軟正黑體", 9),
                                       bg="#f8f9fa", fg="#7f8c8d")
        self.lbl_api_status.pack(anchor="w")

        self.rate_frame = tk.LabelFrame(left_panel, text="🌏 臺銀即時匯率", bg="#f8f9fa", font=("微軟正黑體", 10))
        self.rate_frame.pack(fill="x", pady=5)
        self.lbl_usd = tk.Label(self.rate_frame, text="USD: 載入中...", bg="#f8f9fa", fg="#d35400",
                                font=("Arial", 10, "bold"))
        self.lbl_usd.pack(anchor="w", padx=10)
        self.lbl_jpy = tk.Label(self.rate_frame, text="JPY: 載入中...", bg="#f8f9fa", fg="#d35400",
                                font=("Arial", 10, "bold"))
        self.lbl_jpy.pack(anchor="w", padx=10)

        tk.Button(left_panel, text="呼叫 AI 進行分析", bg="#8e44ad", fg="white", font=("微軟正黑體", 11, "bold"),
                  command=self.run_ai_diagnosis, relief="flat").pack(fill="x", pady=15, ipady=5)

        self.lbl_total = tk.Label(left_panel, text="總資產: $0", font=("微軟正黑體", 14, "bold"), bg="#f8f9fa",
                                  fg="#2c3e50")
        self.lbl_total.pack(side="bottom", pady=20)

        right_panel = tk.Frame(self.root, bg="white")
        right_panel.place(x=350, y=0, width=700, height=700)

        tree_frame = tk.Frame(right_panel)
        tree_frame.pack(fill="both", expand=True, padx=15, pady=15)
        columns = ("id", "date", "category", "item", "amount", "type")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        self.setup_tree_columns()
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)

        self.chart_frame = tk.LabelFrame(right_panel, text="📊 資產視覺化分析", font=("微軟正黑體", 10), bg="white")
        self.chart_frame.pack(fill="both", expand=True, padx=15, pady=15)
        self.canvas = tk.Canvas(self.chart_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def create_input_field(self, parent, label, default, key):
        tk.Label(parent, text=label, bg="#f8f9fa", font=("微軟正黑體", 10)).pack(anchor="w", pady=(2, 0))
        if isinstance(default, list):
            widget = ttk.Combobox(parent, values=default, state="readonly")
            widget.current(0)
        else:
            widget = ttk.Entry(parent)
            if default: widget.insert(0, default)
        widget.pack(fill="x", pady=2)
        setattr(self, f"entry_{key}", widget)

    def setup_tree_columns(self):
        headers = {"date": "日期", "category": "分類", "item": "項目", "amount": "金額(TWD)", "type": "類型"}
        widths = {"date": 100, "category": 80, "item": 180, "amount": 100, "type": 60}
        self.tree.column("id", width=0, stretch=False)
        for col, text in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=widths.get(col, 100), anchor="center" if col != "item" else "w")

    def update_exchange_rates(self):
        rates = SmartTools.get_exchange_rates()
        if "Error" not in rates:
            self.current_rates = rates
            self.root.after(0, lambda: self.lbl_usd.config(text=f"USD: {rates.get('USD')}"))
            self.root.after(0, lambda: self.lbl_jpy.config(text=f"JPY: {rates.get('JPY')}"))
        else:
            err = rates["Error"]
            self.root.after(0, lambda: self.lbl_usd.config(text="USD: 取得失敗"))
            self.root.after(0, lambda: self.lbl_jpy.config(text="JPY: 取得失敗"))
            print("匯率取得失敗原因：", err)

    def set_api_key(self):
        key = simpledialog.askstring("API Key", "請輸入 Google Gemini API Key:\n(若無 Key 則留空，將使用本地運算)")
        if key:
            self.api_key = key
            self.lbl_api_status.config(text="✅ API Key 已設定 (雲端模式)", fg="green")
            messagebox.showinfo("成功", "已切換至雲端 AI 模式！")
        else:
            self.api_key = ""
            self.lbl_api_status.config(text="⚠️ 未設定 (本地模式)", fg="#7f8c8d")

    def run_ai_diagnosis(self):
        summary = self.db.get_summary()
        total_exp = sum(item[1] for item in summary)
        if self.api_key:
            if total_exp == 0:
                messagebox.showinfo("提示", "目前沒有支出紀錄，AI 無法分析。")
                return
            expense_text = self.db.get_recent_records_text()
            self.root.config(cursor="wait")

            def call_api_thread():
                advice = SmartTools.call_gemini_api(self.api_key, expense_text, total_exp)
                self.root.after(0, lambda: self.root.config(cursor=""))
                self.root.after(0, lambda: messagebox.showinfo("🤖 Gemini AI 理財顧問", advice))

            threading.Thread(target=call_api_thread, daemon=True).start()
        else:
            daily_data = self.db.get_daily_expenses()
            prediction_text, _ = SmartTools.local_ai_prediction(daily_data)
            advice = []
            if total_exp > 0:
                for cat, amt in summary:
                    ratio = amt / total_exp
                    if cat == "娛樂" and ratio > 0.3:
                        advice.append("⚠️ 娛樂支出過高 (>30%)")
                    elif cat == "飲食" and ratio > 0.5:
                        advice.append("🍔 飲食佔比過半")
            final_msg = f"{prediction_text}\n\n(提示: 設定 API Key 可獲得真人般的理財建議)\n\n" + (
                "\n".join(advice) if advice else "✅ 消費結構健康")
            messagebox.showinfo("本地 AI 分析", final_msg)

    def add_record(self):
        try:
            date = self.entry_date.get()
            cat = self.entry_category.get()
            item = self.entry_item.get()
            raw_amount = self.entry_amount.get()
            currency = self.combo_currency.get()
            rtype = self.entry_type.get()

            if not item or not raw_amount: raise ValueError

            amount = float(raw_amount)
            final_item_name = item

            # --- 匯率換算邏輯 ---
            if currency != "TWD":
                if currency in self.current_rates:
                    rate = self.current_rates[currency]
                    converted_amount = int(amount * rate)
                    final_item_name = f"{item} ({currency} {amount})"  # 自動備註原幣
                    messagebox.showinfo("匯率換算",
                                        f"匯率: {rate}\n原價: {currency} {amount}\n折合台幣: ${converted_amount}")
                    amount = converted_amount
                else:
                    messagebox.showwarning("匯率警告", f"目前無法取得 {currency} 匯率，將以 1:1 記錄。")

            self.db.add_record(date, cat, final_item_name, int(amount), rtype)
            self.entry_item.delete(0, tk.END)
            self.entry_amount.delete(0, tk.END)
            self.refresh_data()
        except ValueError:
            messagebox.showerror("錯誤", "金額格式錯誤或欄位空白")

    def delete_record(self):
        sel = self.tree.selection()
        if sel and messagebox.askyesno("確認", "刪除此紀錄？"):
            self.db.delete_record(self.tree.item(sel[0])['values'][0])
            self.refresh_data()

    def export_csv(self):
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if f:
            with open(f, 'w', newline='', encoding='utf-8-sig') as file:
                csv.writer(file).writerows([["ID", "日期", "分類", "項目", "金額", "類型"]] + self.db.fetch_all())
            messagebox.showinfo("成功", "匯出完成")

    def refresh_data(self):
        self.tree.delete(*self.tree.get_children())
        recs = self.db.fetch_all()
        asset = sum(r[4] if r[5] == "收入" else -r[4] for r in recs)
        for r in recs: self.tree.insert("", "end", values=r)
        self.lbl_total.config(text=f"總資產: ${asset:,}", fg="#27ae60" if asset >= 0 else "#c0392b")
        self.draw_pie_chart()

    def draw_pie_chart(self):
        self.canvas.delete("all")
        data = self.db.get_summary()
        if not data:
            self.canvas.create_text(250, 200, text="無資料", font=("微軟正黑體", 14), fill="#95a5a6")
            return
        total = sum(d[1] for d in data)
        start, cx, cy, r = 0, 250, 200, 150
        colors = ["#ff7675", "#74b9ff", "#55efc4", "#ffeaa7", "#a29bfe", "#fd79a8", "#00b894", "#fdcb6e"]
        for i, (cat, amt) in enumerate(data):
            extent = (amt / total) * 360
            self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=start, extent=extent, fill=colors[i % 8],
                                   outline="white")
            self.canvas.create_rectangle(500, 50 + i * 30, 520, 70 + i * 30, fill=colors[i % 8], outline="")
            self.canvas.create_text(530, 60 + i * 30, anchor="w", text=f"{cat}: {amt / total:.1%} (${amt})")
            start += extent


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartFinanceApp(root)
    root.mainloop()