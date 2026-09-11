from flask import Flask, jsonify, send_file, request, session, redirect, url_for
import csv
import os

app = Flask(__name__)

# ==============================
# SECURITY SETTINGS
# ==============================

app.secret_key = "CHANGE_THIS_TO_A_RANDOM_SECRET_KEY"

USERNAME = "admin"
PASSWORD = "engine123"

CSV_FILE = "/home/manvith/Downloads/engine_can_sensor_log.csv"


# ==============================
# LOGIN CHECK
# ==============================

def logged_in():
    return session.get("logged_in") is True


# ==============================
# SHARED HUD STYLES
# ==============================

HUD_STYLE = """
    :root {
        --bg: #060a08;
        --panel: #0b1310;
        --line: #123028;
        --accent: #00ff9d;
        --accent-dim: #0a9c66;
        --accent2: #00d9ff;
        --danger: #ff4d4d;
        --text: #d7ffe9;
        --text-dim: #6fa98d;
    }

    * { box-sizing: border-box; }

    @keyframes scan {
        0%   { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.25; }
    }

    @keyframes flicker {
        0%, 96%, 100% { opacity: 1; }
        97% { opacity: 0.7; }
        98% { opacity: 1; }
    }

    body {
        margin: 0;
        min-height: 100vh;
        background:
            linear-gradient(rgba(0,255,157,0.03) 1px, transparent 1px) 0 0 / 100% 26px,
            linear-gradient(90deg, rgba(0,255,157,0.03) 1px, transparent 1px) 0 0 / 26px 100%,
            radial-gradient(ellipse at top, #0d1a15 0%, #060a08 65%);
        color: var(--text);
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        position: relative;
        overflow-x: hidden;
    }

    body::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        opacity: 0.5;
        animation: scan 6s linear infinite;
        pointer-events: none;
        z-index: 999;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 26px;
        border-bottom: 1px solid var(--line);
        background: rgba(11,19,16,0.7);
        letter-spacing: 2px;
        font-size: 12px;
        color: var(--text-dim);
    }

    .topbar .brand {
        color: var(--accent);
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 14px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 8px var(--accent), 0 0 16px var(--accent);
        margin-right: 8px;
        animation: blink 1.6s infinite;
        vertical-align: middle;
    }

    .frame {
        max-width: 960px;
        margin: 40px auto;
        padding: 0 20px;
    }

    .panel {
        position: relative;
        background: linear-gradient(180deg, rgba(0,255,157,0.04), rgba(0,0,0,0) 40%), var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 30px;
        box-shadow: 0 0 40px rgba(0,255,157,0.06), inset 0 0 60px rgba(0,255,157,0.02);
        animation: flicker 7s infinite;
    }

    .panel::before, .panel::after,
    .corner-tl, .corner-tr, .corner-bl, .corner-br {
        content: "";
        position: absolute;
        width: 18px;
        height: 18px;
        border: 2px solid var(--accent);
        opacity: 0.85;
    }
    .corner-tl { top: -1px; left: -1px; border-right: none; border-bottom: none; }
    .corner-tr { top: -1px; right: -1px; border-left: none; border-bottom: none; }
    .corner-bl { bottom: -1px; left: -1px; border-right: none; border-top: none; }
    .corner-br { bottom: -1px; right: -1px; border-left: none; border-top: none; }

    h1, h2 {
        font-family: 'Orbitron', sans-serif;
        color: var(--accent);
        text-shadow: 0 0 10px rgba(0,255,157,0.5);
        letter-spacing: 3px;
        margin-top: 0;
    }

    .sub {
        color: var(--text-dim);
        font-size: 12px;
        letter-spacing: 2px;
        margin-bottom: 24px;
        text-transform: uppercase;
    }

    .crosshair {
        width: 46px;
        height: 46px;
        margin: 0 auto 18px;
        position: relative;
        opacity: 0.9;
    }
    .crosshair::before, .crosshair::after {
        content: "";
        position: absolute;
        background: var(--accent);
    }
    .crosshair::before { top: 50%; left: 0; right: 0; height: 1px; transform: translateY(-50%); }
    .crosshair::after { left: 50%; top: 0; bottom: 0; width: 1px; transform: translateX(-50%); }
    .crosshair .ring {
        position: absolute;
        inset: 6px;
        border: 1px solid var(--accent);
        border-radius: 50%;
    }

    input {
        display: block;
        width: 100%;
        margin: 14px 0;
        padding: 13px 14px;
        border-radius: 3px;
        border: 1px solid var(--line);
        background: #04100b;
        color: var(--accent);
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
        outline: none;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    input::placeholder { color: var(--text-dim); }
    input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 10px rgba(0,255,157,0.35);
    }

    button, .btn {
        width: 100%;
        padding: 13px 20px;
        border: 1px solid var(--accent);
        border-radius: 3px;
        background: rgba(0,255,157,0.08);
        color: var(--accent);
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        letter-spacing: 2px;
        cursor: pointer;
        transition: all 0.2s;
        text-decoration: none;
        display: block;
        text-align: center;
        box-sizing: border-box;
    }
    button:hover, .btn:hover {
        background: var(--accent);
        color: #04100b;
        box-shadow: 0 0 20px rgba(0,255,157,0.6);
    }

    .btn-row { display: grid; gap: 14px; margin-top: 10px; }

    .btn.secondary { border-color: var(--accent2); color: var(--accent2); background: rgba(0,217,255,0.06); }
    .btn.secondary:hover { background: var(--accent2); color: #04100b; box-shadow: 0 0 20px rgba(0,217,255,0.6); }

    .btn.danger { border-color: var(--danger); color: var(--danger); background: rgba(255,77,77,0.06); }
    .btn.danger:hover { background: var(--danger); color: #1a0505; box-shadow: 0 0 20px rgba(255,77,77,0.6); }

    .error {
        color: var(--danger);
        font-size: 12px;
        letter-spacing: 1px;
        text-align: center;
        min-height: 16px;
        text-transform: uppercase;
    }

    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 14px;
        margin: 24px 0 30px;
    }
    .tele-box {
        border: 1px solid var(--line);
        background: rgba(0,255,157,0.03);
        border-radius: 3px;
        padding: 14px;
        text-align: center;
    }
    .tele-box .label {
        font-size: 10px;
        color: var(--text-dim);
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .tele-box .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 20px;
        color: var(--accent2);
        margin-top: 6px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        margin-top: 10px;
    }
    th, td {
        border-bottom: 1px solid var(--line);
        padding: 8px 10px;
        text-align: left;
        white-space: nowrap;
    }
    th {
        color: var(--accent);
        font-family: 'Orbitron', sans-serif;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        position: sticky;
        top: 0;
        background: var(--panel);
    }
    td { color: var(--text); }
    tr:hover td { background: rgba(0,255,157,0.05); }

    .table-wrap { max-height: 460px; overflow: auto; border: 1px solid var(--line); border-radius: 3px; }

    .footer-note {
        margin-top: 20px;
        text-align: center;
        color: var(--text-dim);
        font-size: 11px;
        letter-spacing: 1px;
    }
"""

HUD_HEAD = """
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>%s</style>
""" % HUD_STYLE


# ==============================
# LOGIN PAGE
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():

    error = ""

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        if username == USERNAME and password == PASSWORD:

            session["logged_in"] = True

            return redirect(url_for("home"))

        error = "ACCESS DENIED — INVALID CREDENTIALS"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UAV // Engine Uplink</title>
        {HUD_HEAD}
    </head>
    <body>

        <div class="topbar">
            <div class="brand">◈ UAV-OPS TELEMETRY</div>
            <div><span class="status-dot"></span>LINK STANDBY</div>
        </div>

        <div class="frame" style="max-width:420px;">
            <div class="panel">
                <span class="corner-tl"></span><span class="corner-tr"></span>
                <span class="corner-bl"></span><span class="corner-br"></span>

                <div class="crosshair"><span class="ring"></span></div>

                <h1 style="text-align:center;">ENGINE UPLINK</h1>
                <p class="sub" style="text-align:center;">Authenticate to access flight engine telemetry</p>

                <form method="POST">
                    <input type="text" name="username" placeholder="OPERATOR ID" required autocomplete="off">
                    <input type="password" name="password" placeholder="ACCESS CODE" required>
                    <button type="submit">▶ ESTABLISH LINK</button>
                </form>

                <p class="error">{error}</p>
            </div>
            <p class="footer-note">SECURE CHANNEL · RASPBERRY PI GROUND NODE</p>
        </div>

    </body>
    </html>
    """


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    if not logged_in():
        return redirect(url_for("login"))

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UAV // Engine Dashboard</title>
        {HUD_HEAD}
    </head>
    <body>

        <div class="topbar">
            <div class="brand">◈ UAV-OPS TELEMETRY</div>
            <div><span class="status-dot"></span>LINK ACTIVE — PI ONLINE</div>
        </div>

        <div class="frame">
            <div class="panel">
                <span class="corner-tl"></span><span class="corner-tr"></span>
                <span class="corner-bl"></span><span class="corner-br"></span>

                <h1>ENGINE MONITORING SYSTEM</h1>
                <p class="sub">Ground control interface · CAN bus sensor uplink</p>

                <div class="telemetry-grid">
                    <div class="tele-box">
                        <div class="label">Node</div>
                        <div class="value">RPi-01</div>
                    </div>
                    <div class="tele-box">
                        <div class="label">Link</div>
                        <div class="value" style="color:var(--accent);">ONLINE</div>
                    </div>
                    <div class="tele-box">
                        <div class="label">Bus</div>
                        <div class="value">CAN</div>
                    </div>
                    <div class="tele-box">
                        <div class="label">Mode</div>
                        <div class="value">LOG</div>
                    </div>
                </div>

                <div class="btn-row">
                    <a class="btn secondary" href="/dashboard">▣ VIEW SENSOR TELEMETRY</a>
                    <a class="btn" href="/download">⇩ DOWNLOAD CSV LOG</a>
                    <a class="btn danger" href="/logout">⏻ TERMINATE SESSION</a>
                </div>
            </div>
            <p class="footer-note">ENGINE_CAN_SENSOR_LOG.CSV · GROUND STATION v1.0</p>
        </div>

    </body>
    </html>
    """


# ==============================
# LIVE TELEMETRY DASHBOARD (HTML VIEW OF /data)
# ==============================

@app.route("/dashboard")
def dashboard():

    if not logged_in():
        return redirect(url_for("login"))

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UAV // Sensor Telemetry</title>
        {HUD_HEAD}
    </head>
    <body>

        <div class="topbar">
            <div class="brand">◈ UAV-OPS TELEMETRY</div>
            <div><span class="status-dot"></span><span id="rowcount">0</span> RECORDS · AUTO-REFRESH 5s</div>
        </div>

        <div class="frame" style="max-width: 1100px;">
            <div class="panel">
                <span class="corner-tl"></span><span class="corner-tr"></span>
                <span class="corner-bl"></span><span class="corner-br"></span>

                <h1>SENSOR TELEMETRY</h1>
                <p class="sub">Live feed · engine_can_sensor_log.csv</p>

                <div class="table-wrap">
                    <table id="tele-table">
                        <thead><tr id="tele-head"></tr></thead>
                        <tbody id="tele-body">
                            <tr><td style="color:var(--text-dim);">▸ AWAITING DATA...</td></tr>
                        </tbody>
                    </table>
                </div>

                <div class="btn-row">
                    <a class="btn secondary" href="/">◂ RETURN TO DASHBOARD</a>
                </div>
            </div>
        </div>

        <script>
            async function refresh() {{
                try {{
                    const res = await fetch('/data');
                    if (!res.ok) throw new Error('no data');
                    const rows = await res.json();

                    const head = document.getElementById('tele-head');
                    const body = document.getElementById('tele-body');
                    document.getElementById('rowcount').textContent = rows.length;

                    if (!rows.length) {{
                        body.innerHTML = '<tr><td style="color:var(--text-dim);">▸ NO RECORDS FOUND</td></tr>';
                        return;
                    }}

                    const cols = Object.keys(rows[0]);
                    head.innerHTML = cols.map(c => `<th>${{c}}</th>`).join('');

                    body.innerHTML = rows.slice(-200).reverse().map(r =>
                        '<tr>' + cols.map(c => `<td>${{r[c] ?? ''}}</td>`).join('') + '</tr>'
                    ).join('');
                }} catch (e) {{
                    document.getElementById('tele-body').innerHTML =
                        '<tr><td style="color:var(--danger);">▸ SIGNAL LOST — RETRYING...</td></tr>';
                }}
            }}

            refresh();
            setInterval(refresh, 5000);
        </script>

    </body>
    </html>
    """


# ==============================
# SENSOR DATA (JSON API)
# ==============================

@app.route("/data")
def data():

    if not logged_in():
        return redirect(url_for("login"))

    if not os.path.exists(CSV_FILE):

        return jsonify({
            "error": "CSV file not found"
        }), 404

    records = []

    with open(CSV_FILE, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:
            records.append(row)

    return jsonify(records)


# ==============================
# DOWNLOAD CSV
# ==============================

@app.route("/download")
def download():

    if not logged_in():
        return redirect(url_for("login"))

    if not os.path.exists(CSV_FILE):
        return "CSV file not found", 404

    return send_file(
        CSV_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name="engine_can_sensor_log.csv"
    )


# ==============================
# LOGOUT
# ==============================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    print("==============================")
    print(" ENGINE MONITORING SERVER")
    print("==============================")
    print("Server: http://0.0.0.0:5000")
    print("==============================")

    app.run(
        host="0.0.0.0",
        port=5000
    )
