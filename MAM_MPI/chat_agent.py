"""
Desktop Agent —— 能操作你电脑的 AI Agent (Web 版)
使用: streamlit run chat_agent.py

支持：读文件、写文件、运行命令、搜索文件、搜索内容
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false
import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import json, os, subprocess, glob as glob_mod, datetime, traceback
from typing import Any

# ==================== 工具定义 ====================
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。可以指定起止行号来只读部分内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（绝对或相对路径）"},
                    "start_line": {"type": "integer", "description": "起始行号(1-based)，不填则从头"},
                    "end_line": {"type": "integer", "description": "结束行号(1-based)，不填则到末尾"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建或覆盖写入文件。安全模式下此操作会被跳过。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出目录中的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认当前工作目录"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "执行 Shell 命令并返回输出。安全模式下此操作会被跳过。禁止执行 rm -rf、格式化等破坏性命令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "working_dir": {"type": "string", "description": "工作目录，可选"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "按文件名模式搜索文件，支持 glob 通配符（如 **/*.py, *.txt）",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式"},
                    "directory": {"type": "string", "description": "搜索目录，默认当前目录"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "在文件内容中搜索文本（类似 grep）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的文本"},
                    "path": {"type": "string", "description": "搜索路径，默认当前目录"},
                    "file_pattern": {"type": "string", "description": "限定文件类型，如 *.py"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "获取当前工作目录的绝对路径",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

DANGEROUS_COMMANDS = [
    "rm -rf /", "format ", "del /f /s /q C:", "rd /s /q C:",
    "shutdown", "mkfs", "diskpart", "> /dev/sda",
]

# ==================== 工具执行函数 ====================
def _safe_path(p: str) -> str:
    """将相对路径转为绝对路径"""
    if not os.path.isabs(p):
        return os.path.abspath(os.path.join(os.getcwd(), p))
    return os.path.abspath(p)


def execute_tool(name: str, args: dict[str, Any], safe_mode: bool) -> str:
    """执行工具，返回结果字符串。safe_mode 下跳过写和命令操作。"""
    cwd = os.getcwd()

    if name == "read_file":
        p = _safe_path(args["path"])
        if not os.path.exists(p):
            return f"❌ 文件不存在: {p}"
        if not os.path.isfile(p):
            return f"❌ 不是文件: {p}"
        if os.path.getsize(p) > 5 * 1024 * 1024:
            return f"❌ 文件过大 ({os.path.getsize(p)/1024/1024:.0f}MB)，请用 start_line/end_line 分段读取"
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start = max(0, args.get("start_line", 1) - 1)
            end = min(len(lines), args.get("end_line", len(lines)))
            selected = lines[start:end]
            result = "".join(f"{i+start+1:5d}| {l}" for i, l in enumerate(selected))
            if not result.strip():
                return "(空文件)"
            if len(result) > 6000:
                return result[:6000] + f"\n... (截断，共 {len(lines)} 行，{end-start} 行已显示)"
            return result
        except Exception as e:
            return f"❌ 读取失败: {e}"

    elif name == "write_file":
        if safe_mode:
            return f"🛡️ [安全模式] 已跳过写入: {args['path']}\n内容预览:\n{args['content'][:300]}..."
        p = _safe_path(args["path"])
        content = args["content"]
        try:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ 已写入: {p} ({len(content)} 字符, {content.count(chr(10))+1} 行)"
        except Exception as e:
            return f"❌ 写入失败: {e}"

    elif name == "list_directory":
        p = _safe_path(args.get("path") or cwd)
        if not os.path.exists(p):
            return f"❌ 目录不存在: {p}"
        if not os.path.isdir(p):
            return f"❌ 不是目录: {p}"
        try:
            items = sorted(os.listdir(p))
            lines = []
            for item in items:
                full = os.path.join(p, item)
                size_str = ""
                if os.path.isfile(full):
                    s = os.path.getsize(full)
                    size_str = f"  [{s/1024:6.1f} KB]"
                tag = "📁" if os.path.isdir(full) else "📄"
                lines.append(f"  {tag} {item}{size_str}")
            header = f"📂 {p}\n"
            body = "\n".join(lines[:80])
            footer = f"\n  ... (共 {len(items)} 项)" if len(items) > 80 else ""
            return header + body + footer
        except Exception as e:
            return f"❌ 列目录失败: {e}"

    elif name == "run_shell":
        if safe_mode:
            return f"🛡️ [安全模式] 已跳过命令: {args['command']}"
        cmd = args["command"]
        wd = args.get("working_dir") or cwd
        if any(d in cmd.lower() for d in DANGEROUS_COMMANDS):
            return f"⛔ 命令被拒绝（包含危险操作）: {cmd}"
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=wd,
                capture_output=True, text=True,
                timeout=120,
                encoding="utf-8", errors="replace",
            )
            out = result.stdout.strip()[:4000]
            err = result.stderr.strip()[:2000]
            parts = []
            if out:
                parts.append(out)
            if err:
                parts.append(f"[stderr]\n{err}")
            if not parts:
                parts.append("(无输出)")
            return f"退出码: {result.returncode}\n" + "\n".join(parts)
        except subprocess.TimeoutExpired:
            return "⏱️ 命令超时 (120s)"
        except Exception as e:
            return f"❌ 执行失败: {e}"

    elif name == "search_files":
        pattern = args["pattern"]
        directory = _safe_path(args.get("directory") or cwd)
        try:
            search_path = os.path.join(directory, pattern)
            matches = sorted(glob_mod.glob(search_path, recursive=True))[:60]
            if not matches:
                return f"未找到匹配 '{pattern}' 的文件"
            return f"🔍 找到 {len(matches)} 个文件:\n" + "\n".join(f"  📄 {m}" for m in matches)
        except Exception as e:
            return f"❌ 搜索失败: {e}"

    elif name == "search_content":
        query = args["query"]
        spath = _safe_path(args.get("path") or cwd)
        fp = args.get("file_pattern") or "*"
        try:
            results = []
            count = 0
            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
            for root, dirs, files in os.walk(spath):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                for fname in files:
                    if not glob_mod.fnmatch.fnmatch(fname, fp):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        if os.path.getsize(fpath) > 2 * 1024 * 1024:
                            continue
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            for lineno, line in enumerate(f, 1):
                                if query.lower() in line.lower():
                                    results.append(f"  {fpath}:{lineno}: {line.strip()[:150]}")
                                    count += 1
                                    if count >= 40:
                                        raise StopIteration
                    except Exception:
                        pass
        except StopIteration:
            pass
        except Exception as e:
            return f"❌ 搜索失败: {e}"
        if not results:
            return f"未找到包含 '{query}' 的文件"
        return f"🔍 搜索 '{query}' 结果:\n" + "\n".join(results)

    elif name == "get_working_directory":
        return f"当前工作目录: {cwd}"

    return f"❌ 未知工具: {name}"


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="Desktop Agent",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🖥️ Desktop Agent")
    st.caption("连接远程混元 · 操作本地电脑")

    st.header("🔌 API 连接")
    api_base = st.text_input("Base URL", value="http://172.31.1.10:8080/v1")
    api_key = st.text_input("API Key", value="sk-hy3-admin-001", type="password")
    model_name = st.text_input("Model", value="hy3-preview")

    st.header("🎛️ 参数")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 16384, 4096, 256)

    st.header("🛡️ 安全")
    safe_mode = st.checkbox(
        "🛡️ 安全模式",
        value=False,
        help="开启后：write_file 和 run_shell 将被跳过，只读不写",
    )

    st.header("💬 会话")
    col1, col2 = st.columns(2)
    if col1.button("🔄 新建对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_logs = []
        st.rerun()
    if col2.button("🗑️ 清除日志", use_container_width=True):
        st.session_state.tool_logs = []
        st.rerun()

    st.divider()
    st.caption(f"📁 {os.getcwd()}")

# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_logs" not in st.session_state:
    st.session_state.tool_logs: list[dict[str, Any]] = []

# ==================== 客户端 ====================
@st.cache_resource
def get_client(base_url: str, api_key: str):
    return OpenAI(base_url=base_url, api_key=api_key)

try:
    client = get_client(api_base, api_key)
except Exception as e:
    st.error(f"客户端初始化失败: {e}")
    st.stop()

# ==================== Agent 循环 ====================
def run_agent(user_input: str):
    """运行 Agent 循环"""
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # System prompt（动态注入工作目录）
    system_prompt = f"""你是桌面 AI Agent，运行在 Windows 11 上，可以通过工具直接操作电脑。

## 你的工具
| 工具 | 用途 | 限制 |
|------|------|------|
| read_file | 读取文件 | 单次最多 5MB |
| write_file | 创建/覆盖文件 | {"安全模式下禁用" if safe_mode else "可用"} |
| list_directory | 列出目录 | - |
| run_shell | 执行命令 | {"安全模式下禁用" if safe_mode else "可用，禁止危险命令"} |
| search_files | 按文件名搜索 | glob 模式 |
| search_content | 搜索文件内容 | grep 风格 |
| get_working_directory | 获取当前目录 | - |

## 工作原则
1. **直接行动**：不要只给建议，用工具去执行。用户说"创建文件"你就 write_file。
2. **先查后改**：修改文件前先用 read_file 查看内容。
3. **一次到位**：尽量在一次回复中完成完整任务。
4. **简洁报告**：做完后简短说明结果，不要啰嗦。

## 当前状态
- 工作目录: {os.getcwd()}
- 时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 安全模式: {"开启 (只读)" if safe_mode else "关闭 (可读写)"}"""

    full_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        *st.session_state.messages,
    ]

    # Agent 循环：模型 ↔ 工具执行
    status_container = st.empty()
    for turn in range(8):
        status_container.info(f"🤔 第 {turn+1} 轮...")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=TOOLS,
                tool_choice="auto",
                stream=False,
            )
        except Exception as e:
            status_container.empty()
            with st.chat_message("assistant"):
                st.error(f"API 错误: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"[API 错误] {e}"})
            return

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        # --- 工具调用 ---
        if finish == "tool_calls" and msg.tool_calls:
            # 保存助手消息
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            st.session_state.messages.append(assistant_msg)  # type: ignore[arg-type]
            full_messages.append(assistant_msg)  # type: ignore[arg-type]

            # 执行每个工具
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args: dict[str, Any] = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result_text = execute_tool(name, args, safe_mode)

                # 记录日志
                log_entry: dict[str, Any] = {
                    "turn": turn + 1,
                    "tool": name,
                    "args": args,
                    "result": result_text[:1500],
                }
                st.session_state.tool_logs.append(log_entry)

                # 展示工具调用（可折叠）
                with st.chat_message("assistant"):
                    emoji_map = {
                        "read_file": "📖", "write_file": "✍️", "list_directory": "📂",
                        "run_shell": "💻", "search_files": "🔍", "search_content": "🔎",
                        "get_working_directory": "📁",
                    }
                    emoji = emoji_map.get(name, "🔧")
                    safe_badge = " 🛡️已跳过" if "安全模式" in result_text else ""

                    with st.expander(f"{emoji} {name}{safe_badge} — {_summarize_args(name, args)}", expanded=False):
                        st.caption("**参数**")
                        st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
                        st.caption("**结果**")
                        if "错误" in result_text or "失败" in result_text:
                            st.error(result_text[:2000])
                        elif "安全模式" in result_text:
                            st.warning(result_text[:2000])
                        else:
                            st.code(result_text[:2000], language="text" if result_text[0].isdigit() else None)

                # 添加工具结果
                tool_msg: dict[str, Any] = {"role": "tool", "content": result_text, "tool_call_id": tc.id}
                st.session_state.messages.append(tool_msg)  # type: ignore[arg-type]
                full_messages.append(tool_msg)  # type: ignore[arg-type]

        # --- 文本回复 ---
        elif msg.content:
            status_container.empty()
            st.session_state.messages.append({"role": "assistant", "content": msg.content})
            with st.chat_message("assistant"):
                st.markdown(msg.content)
            return

        # --- 其他 ---
        else:
            status_container.empty()
            fallback = f"(结束: {finish})"
            st.session_state.messages.append({"role": "assistant", "content": fallback})
            return

    status_container.empty()
    timeout_msg = "⏱️ 达到最大循环次数，任务可能未完成。请尝试分解为更小的步骤。"
    st.session_state.messages.append({"role": "assistant", "content": timeout_msg})
    with st.chat_message("assistant"):
        st.warning(timeout_msg)


def _summarize_args(name: str, args: dict[str, Any]) -> str:
    """生成工具调用的简短描述"""
    if name == "read_file":
        p = args.get("path", "?")
        return os.path.basename(p)
    if name == "write_file":
        p = args.get("path", "?")
        return os.path.basename(p)
    if name == "list_directory":
        return args.get("path") or "."
    if name == "run_shell":
        cmd = args.get("command", "?")
        return cmd[:60] + ("..." if len(cmd) > 60 else "")
    if name == "search_files":
        return args.get("pattern", "?")
    if name == "search_content":
        return args.get("query", "?")
    return ""


# ==================== 主界面 ====================
st.title("🖥️ Desktop Agent")

# 欢迎
if not st.session_state.messages:
    st.info("""
    👋 **我是你的桌面 AI Agent，能直接操作这台电脑！**

    | 能力 | 示例 |
    |------|------|
    | 📖 读文件 | "看看 chat_agent.py 里有什么" |
    | ✍️ 写文件 | "创建一个 test.py 打印 Hello" |
    | 💻 运行命令 | "运行 pip list 看看安装了哪些包" |
    | 🔍 搜索 | "搜索所有包含 'TODO' 的 Python 文件" |
    | 📂 浏览目录 | "列出当前目录的所有文件" |
    """)

# 重新渲染历史消息（跳过 tool 消息和 tool_calls 消息）
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    if role == "tool":
        continue
    if role == "assistant" and msg.get("tool_calls"):
        continue
    if role in ("user", "assistant"):
        content = msg.get("content", "")
        if content and not content.startswith("(结束"):
            with st.chat_message(role):
                st.markdown(content)

# 用户输入
if prompt := st.chat_input("告诉我想做什么，我会直接用工具完成..."):
    run_agent(prompt)

# 侧边栏：工具调用历史
with st.sidebar:
    st.divider()
    st.caption(f"📋 工具调用日志 ({len(st.session_state.tool_logs)} 次)")
    if st.session_state.tool_logs:
        for log in reversed(st.session_state.tool_logs[-10:]):
            emoji = {
                "read_file": "📖", "write_file": "✍️", "list_directory": "📂",
                "run_shell": "💻", "search_files": "🔍", "search_content": "🔎",
                "get_working_directory": "📁",
            }.get(log["tool"], "🔧")
            st.caption(f"{emoji} {log['tool']} → {_summarize_args(log['tool'], log['args'])}")
