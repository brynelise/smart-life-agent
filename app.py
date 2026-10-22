# -*- coding: utf-8 -*-
"""
多 Agent 协同的智慧生活服务助手
基于 spec.md / design.md / tasks.md 规格文档实现
技术栈: Python + Streamlit
架构: 接入层 / 核心编排层 / 能力支撑层 / 基础设施层
"""

import re
import os
import json
import uuid
import time
import hashlib
import threading
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st


def _load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass
    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and val:
                            config.setdefault(key, val)
        except Exception:
            pass
    try:
        for key in ("llm_api_key", "llm_base_url", "llm_model",
                    "amap_api_key", "amap_base_url"):
            val = st.secrets.get(key)
            if val:
                config.setdefault(key, val)
    except Exception:
        pass
    return config


_CONFIG = _load_config()
AMAP_API_KEY = _CONFIG.get("amap_api_key", "") or os.environ.get("AMAP_API_KEY", "")
AMAP_BASE_URL = _CONFIG.get("amap_base_url", "https://restapi.amap.com/v3")
LLM_API_KEY = (_CONFIG.get("llm_api_key", "")
               or os.environ.get("DEEPSEEK_API_KEY", "")
               or os.environ.get("LLM_API_KEY", ""))
LLM_BASE_URL = (_CONFIG.get("llm_base_url", "https://api.deepseek.com/v1")
                or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"))
LLM_MODEL = (_CONFIG.get("llm_model", "deepseek-chat")
             or os.environ.get("LLM_MODEL", "deepseek-chat"))


def _call_llm(query: str, system_prompt: str, max_tokens: int = 1024) -> str:
    if not LLM_API_KEY:
        print(f"[LLM] ⚠️ API Key 未配置，请检查 config.json 中的 llm_api_key 字段")
        return ""
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            print(f"[LLM] ⚠️ 响应无 choices: {data}")
        else:
            error_body = resp.text[:500]
            print(f"[LLM] ❌ API 调用失败: HTTP {resp.status_code} — {error_body}")
    except requests.exceptions.Timeout:
        print(f"[LLM] ❌ 请求超时 (30s)，URL: {url}")
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] ❌ 连接失败: {e}")
    except Exception as e:
        print(f"[LLM] ❌ 未知异常: {type(e).__name__}: {e}")
    return ""


# ==========================================================================
# 一、枚举定义
# ==========================================================================

class ServiceDomain(Enum):
    SCHEDULE = "日程管理"
    HEALTH = "健康顾问"
    HOME = "智能家居"
    SHOPPING = "购物助手"
    SEARCH = "信息检索"
    ROUTE = "路线规划"
    UNKNOWN = "未知领域"


class PermissionLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class TaskStatus(Enum):
    PENDING = "待执行"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    PARTIAL = "部分完成"
    FAILED = "失败"
    CANCELLED = "已取消"


class AgentStatus(Enum):
    ENABLED = "启用"
    DISABLED = "禁用"


class ApprovalResult(Enum):
    PENDING = "待审批"
    APPROVED = "已通过"
    REJECTED = "已拒绝"
    TIMEOUT = "已超时"


class ToolType(Enum):
    QUERY = "查询"
    CONTROL = "控制"
    NOTIFY = "通知"
    DATA_PROCESSING = "数据处理"


# ==========================================================================
# 二、数据模型
# ==========================================================================

@dataclass
class User:
    user_id: str
    name: str = "默认用户"
    permission_level: PermissionLevel = PermissionLevel.MEDIUM
    preferences: dict = field(default_factory=dict)

    def get_preference(self, domain: ServiceDomain) -> dict:
        return self.preferences.get(domain.value, {})

    def set_preference(self, domain: ServiceDomain, pref: dict) -> None:
        self.preferences[domain.value] = pref


@dataclass
class DialogTurn:
    turn_id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: str = ""
    agent_used: str = ""


@dataclass
class SessionContext:
    dialog_history: list = field(default_factory=list)
    recognized_intents: list = field(default_factory=list)
    intermediate_results: dict = field(default_factory=dict)
    referenced_entities: dict = field(default_factory=dict)
    max_capacity: int = 128

    def add_turn(self, turn: DialogTurn) -> None:
        self.dialog_history.append(turn)
        if len(self.dialog_history) > self.max_capacity:
            self.dialog_history.pop(0)

    def add_result(self, key: str, value) -> None:
        self.intermediate_results[key] = value
        if len(self.intermediate_results) > self.max_capacity:
            oldest = next(iter(self.intermediate_results))
            del self.intermediate_results[oldest]

    def get_entity(self, ref: str):
        return self.referenced_entities.get(ref)


@dataclass
class Session:
    session_id: str
    user_id: str
    status: str = "进行中"
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    context: SessionContext = field(default_factory=SessionContext)

    def update_activity(self) -> None:
        self.last_active_at = datetime.now()

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.last_active_at > timedelta(minutes=timeout_minutes)

    def close(self) -> None:
        self.status = "已结束"


@dataclass
class Intent:
    domain: ServiceDomain
    query: str
    slots: dict = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class IntentResult:
    intents: list = field(default_factory=list)
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_question: str = ""


@dataclass
class SubTaskResult:
    sub_task_id: str
    agent_id: str
    success: bool
    response: str
    data: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class SubTask:
    sub_task_id: str
    target_agent_id: str
    intent: Intent
    status: TaskStatus = TaskStatus.PENDING
    result: SubTaskResult = None
    depends_on: list = field(default_factory=list)

    def can_execute(self, completed_ids: set) -> bool:
        return all(dep in completed_ids for dep in self.depends_on)


@dataclass
class OrchestrationTask:
    task_id: str
    session_id: str
    intents: list
    sub_tasks: list = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    shared_context: dict = field(default_factory=dict)
    audit_trail: list = field(default_factory=list)

    def add_sub_task(self, sub_task: SubTask) -> None:
        self.sub_tasks.append(sub_task)

    def get_ready_tasks(self, completed_ids: set) -> list:
        return [st for st in self.sub_tasks
                if st.status == TaskStatus.PENDING and st.can_execute(completed_ids)]


@dataclass
class Tool:
    tool_id: str
    name: str
    tool_type: ToolType
    is_sensitive: bool = False
    timeout_seconds: int = 10
    max_retries: int = 3
    param_spec: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    data: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class ApprovalRecord:
    approval_id: str
    operation: str
    initiated_at: datetime = field(default_factory=datetime.now)
    approver_id: str = ""
    result: ApprovalResult = ApprovalResult.PENDING
    approved_at: datetime = None
    comment: str = ""


@dataclass
class AuditLog:
    log_id: str
    operator: str
    operation_type: str
    target: str
    result: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


@dataclass
class AgentDescriptor:
    agent_id: str
    name: str
    domain: ServiceDomain
    version: str
    capabilities: list
    allowed_tools: list
    permission_level: PermissionLevel
    status: AgentStatus


# ==========================================================================
# 三、安全管控层 (Security & Governance)
# ==========================================================================

class InputSanitizer:
    """输入安全过滤 — spec.md §5.1.3 / §4.3.5"""

    INJECTION_PATTERNS = [
        r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)",
        r"(?i)(exec\s*\(|eval\s*\(|system\s*\()",
        r"(?i)(<script|javascript:|onerror\s*=)",
        r"(?i)(rm\s+-rf|chmod\s+777|sudo\s+)",
        r"(?i)(ignore\s+previous|disregard\s+all|forget\s+instructions)",
    ]
    MAX_INPUT_LENGTH = 4096

    @classmethod
    def sanitize(cls, text: str) -> tuple:
        if not text or not text.strip():
            return False, "输入为空，请重新输入"

        if len(text) > cls.MAX_INPUT_LENGTH:
            return False, f"输入内容过长（超过{cls.MAX_INPUT_LENGTH}字符），请精简后重新发送"

        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text):
                return False, "输入包含不合规内容，请求已被拒绝"

        cleaned = re.sub(r'[<>"\']', '', text).strip()
        return True, cleaned


class PrivacyCipher:
    """隐私数据加密 — spec.md §4.3.2"""

    @staticmethod
    def encrypt(data: str, user_id: str) -> str:
        salt = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return hashlib.sha256((data + salt).encode()).hexdigest()

    @staticmethod
    def mask(data: str, visible_chars: int = 2) -> str:
        if len(data) <= visible_chars:
            return "*" * len(data)
        return data[:visible_chars] + "*" * (len(data) - visible_chars)


class AuditLogger:
    """审计日志 — spec.md §4.4.2 / §5.6.1.4"""

    def __init__(self):
        self._logs: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

    def log(self, operator: str, operation_type: str, target: str,
            result: str, details: dict = None) -> AuditLog:
        entry = AuditLog(
            log_id=str(uuid.uuid4())[:8],
            operator=operator,
            operation_type=operation_type,
            target=target,
            result=result,
            details=details or {}
        )
        with self._lock:
            self._logs.append(entry)
        return entry

    def get_logs(self, limit: int = 100, filter_type: str = None) -> list:
        with self._lock:
            logs = list(self._logs)
        if filter_type:
            logs = [l for l in logs if l.operation_type == filter_type]
        return logs[-limit:]


class ApprovalManager:
    """敏感操作审批管理 — spec.md §5.6.1.2"""

    def __init__(self, audit_logger: AuditLogger):
        self._pending: dict = {}
        self._audit = audit_logger
        self._timeout_seconds = 300

    def request_approval(self, operation: str, approver_id: str) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=str(uuid.uuid4())[:8],
            operation=operation,
            approver_id=approver_id
        )
        self._pending[record.approval_id] = record
        self._audit.log(approver_id, "审批发起", operation, "待审批",
                        {"approval_id": record.approval_id})
        return record

    def process_approval(self, approval_id: str, approved: bool,
                         comment: str = "") -> ApprovalRecord:
        record = self._pending.get(approval_id)
        if not record:
            return None
        record.result = ApprovalResult.APPROVED if approved else ApprovalResult.REJECTED
        record.approved_at = datetime.now()
        record.comment = comment
        self._audit.log(record.approver_id, "审批处理", record.operation,
                        "通过" if approved else "拒绝", {"comment": comment})
        return record

    def auto_approve(self, approval_id: str) -> ApprovalRecord:
        return self.process_approval(approval_id, True, "自动审批通过")

    def get_pending(self) -> list:
        return [r for r in self._pending.values()
                if r.result == ApprovalResult.PENDING]


class PermissionChecker:
    """权限分级校验 — spec.md §5.6.1.1 / §4.3.4"""

    @staticmethod
    def check_agent_permission(agent_level: PermissionLevel,
                               task_sensitivity: PermissionLevel) -> bool:
        return agent_level.value >= task_sensitivity.value

    @staticmethod
    def check_tool_authorization(agent_allowed_tools: list, tool_id: str) -> bool:
        return tool_id in agent_allowed_tools


# ==========================================================================
# 四、上下文管理 (Context Management)
# ==========================================================================

class ContextManager:
    """会话上下文 + 跨 Agent 共享 + 用户偏好 — spec.md §5.4"""

    def __init__(self):
        self._user_preferences: dict = {}
        self._lock = threading.Lock()

    def save_preference(self, user_id: str, domain: ServiceDomain, pref: dict) -> None:
        with self._lock:
            if user_id not in self._user_preferences:
                self._user_preferences[user_id] = {}
            self._user_preferences[user_id][domain.value] = pref

    def get_preference(self, user_id: str, domain: ServiceDomain) -> dict:
        with self._lock:
            return self._user_preferences.get(user_id, {}).get(domain.value, {})

    def get_all_preferences(self, user_id: str) -> dict:
        with self._lock:
            return self._user_preferences.get(user_id, {})

    @staticmethod
    def update_shared_context(task: OrchestrationTask, key: str, value) -> None:
        task.shared_context[key] = value

    @staticmethod
    def get_shared_context(task: OrchestrationTask, key: str):
        return task.shared_context.get(key)


# ==========================================================================
# 五、工具调用框架 (Tool Invocation Framework)
# ==========================================================================

class ToolRegistry:
    """工具注册表 — spec.md §5.5 / §6.5"""

    def __init__(self):
        self._tools: dict = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> Tool:
        return self._tools.get(tool_id)

    def list_tools(self) -> list:
        return list(self._tools.values())


class ToolExecutor:
    """工具执行器（超时/重试）— spec.md §5.5.1.3"""

    def __init__(self, registry: ToolRegistry, audit: AuditLogger):
        self._registry = registry
        self._audit = audit

    def invoke(self, agent_id: str, tool_id: str, params: dict) -> ToolResult:
        tool = self._registry.get(tool_id)
        if not tool:
            self._audit.log(agent_id, "工具调用", tool_id, "失败-未注册")
            return ToolResult(False, error=f"工具 {tool_id} 未注册")

        for attempt in range(tool.max_retries + 1):
            try:
                result = self._execute_tool(tool, params)
                self._audit.log(agent_id, "工具调用", tool_id, "成功",
                                {"attempt": attempt + 1, "params": params})
                return result
            except Exception as e:
                if attempt < tool.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                self._audit.log(agent_id, "工具调用", tool_id, f"失败-{str(e)}",
                                {"attempts": attempt + 1})
                return ToolResult(False, error=str(e))

    def _execute_tool(self, tool: Tool, params: dict) -> ToolResult:
        if tool.tool_id == "weather_query":
            return self._amap_weather(params)
        elif tool.tool_id == "device_control":
            device = params.get("device", "")
            action = params.get("action", "")
            if action in ("打开", "open"):
                VirtualDeviceState.apply_command({"device": device, "action": "open"})
            elif action in ("关闭", "close"):
                VirtualDeviceState.apply_command({"device": device, "action": "close"})
            return ToolResult(True, data={"device": device, "action": action,
                                          "status": "已执行",
                                          "current_state": VirtualDeviceState.get_state()})
        elif tool.tool_id == "calendar_query":
            date_str = params.get("date", "今天")
            resolved = ScheduleStore.resolve_date(date_str)
            events = ScheduleStore.query(resolved)
            return ToolResult(True, data={"events": events, "date": date_str,
                                          "resolved_date": resolved})
        elif tool.tool_id == "calendar_add":
            date_str = params.get("date", "今天")
            time_str = params.get("time", "")
            event = params.get("event", "")
            if not event:
                return ToolResult(False, error="事件内容为空")
            resolved = ScheduleStore.resolve_date(date_str)
            entry = ScheduleStore.add(resolved, time_str, event)
            return ToolResult(True, data={"entry": entry, "date": date_str,
                                          "resolved_date": resolved})
        elif tool.tool_id == "product_search":
            keyword = params.get("keyword", "")
            return ToolResult(True, data={"keyword": keyword,
                                          "products": [],
                                          "message": "未接入真实电商平台API，请配置 LLM 以获取智能推荐"})
        elif tool.tool_id == "health_query":
            topic = params.get("topic", "general")
            return ToolResult(True, data={"topic": topic,
                                          "steps": 8532, "heart_rate": 72,
                                          "sleep": "7.5h"})
        elif tool.tool_id == "amap_geocode":
            return self._amap_geocode(params)
        elif tool.tool_id == "amap_direction_driving":
            return self._amap_direction(params, "driving")
        elif tool.tool_id == "amap_direction_walking":
            return self._amap_direction(params, "walking")
        elif tool.tool_id == "amap_direction_transit":
            return self._amap_direction(params, "transit")
        elif tool.tool_id == "amap_distance":
            return self._amap_distance(params)
        return ToolResult(True, data={"message": "工具执行完成"})

    @staticmethod
    def _amap_request(url: str, params: dict) -> dict:
        if not AMAP_API_KEY:
            return {"status": "0", "info": "NO_API_KEY"}
        params["key"] = AMAP_API_KEY
        try:
            resp = requests.get(url, params=params, timeout=10)
            return resp.json()
        except Exception as e:
            return {"status": "0", "info": str(e)}

    def _amap_geocode(self, params: dict) -> ToolResult:
        address = params.get("address", "")
        if not address:
            return ToolResult(False, error="地址参数为空")
        url = f"{AMAP_BASE_URL}/geocode/geo"
        result = self._amap_request(url, {"address": address})
        if result.get("status") == "1" and result.get("geocodes"):
            geo = result["geocodes"][0]
            return ToolResult(True, data={
                "location": geo.get("location", ""),
                "formatted_address": geo.get("formatted_address", address),
                "province": geo.get("province", ""),
                "city": geo.get("city", ""),
                "adcode": geo.get("adcode", ""),
                "lng": geo.get("location", "").split(",")[0] if geo.get("location") else "",
                "lat": geo.get("location", "").split(",")[1] if geo.get("location") else "",
            })
        info = result.get("info", "未知错误")
        return ToolResult(False, error=f"地理编码失败：{info}（地址：{address}）")

    def _amap_direction(self, params: dict, mode: str) -> ToolResult:
        origin = params.get("origin", "")
        destination = params.get("destination", "")
        strategy = params.get("strategy", "")
        if not origin or not destination:
            return ToolResult(False, error="起点或终点参数为空")
        if "," not in origin:
            geo = self._amap_geocode({"address": origin})
            if geo.success:
                origin = geo.data.get("location", origin)
            else:
                return ToolResult(False, error=f"无法解析起点地址「{origin}」：{geo.error}")
        if "," not in destination:
            geo = self._amap_geocode({"address": destination})
            if geo.success:
                destination = geo.data.get("location", destination)
            else:
                return ToolResult(False, error=f"无法解析终点地址「{destination}」：{geo.error}")
        url = f"{AMAP_BASE_URL}/direction/{mode}"
        req_params = {"origin": origin, "destination": destination,
                       "extensions": "all", "output": "json"}
        if mode == "driving" and strategy:
            req_params["strategy"] = strategy
        elif mode == "transit":
            req_params["city"] = params.get("city", "北京")
        result = self._amap_request(url, req_params)
        if result.get("status") == "1":
            route_data = self._parse_amap_route(result, mode)
            return ToolResult(True, data=route_data)
        mode_names = {"driving": "驾车", "walking": "步行", "transit": "公交"}
        info = result.get("info", "未知错误")
        return ToolResult(False, error=f"{mode_names.get(mode, mode)}路线规划失败：{info}")

    @staticmethod
    def _parse_amap_route(result: dict, mode: str) -> dict:
        mode_names = {"driving": "驾车", "walking": "步行", "transit": "公交"}
        route_info = {"mode": mode_names.get(mode, mode)}
        try:
            if mode == "driving" and "route" in result:
                path = result["route"]["paths"][0]
                route_info["distance"] = f"{float(path['distance'])/1000:.1f}公里"
                route_info["duration"] = f"{int(path['duration'])//60}分钟"
                route_info["steps"] = [s["instruction"] for s in path["steps"]]
            elif mode == "walking" and "route" in result:
                path = result["route"]["paths"][0]
                route_info["distance"] = f"{float(path['distance'])/1000:.1f}公里"
                route_info["duration"] = f"{int(path['duration'])//60}分钟"
                route_info["steps"] = [s["instruction"] for s in path["steps"]]
            elif mode == "transit" and "route" in result:
                transits = result["route"].get("transits", [])
                if transits:
                    t = transits[0]
                    route_info["distance"] = f"{float(t['distance'])/1000:.1f}公里"
                    route_info["duration"] = f"{int(t['duration'])//60}分钟"
                    route_info["steps"] = [seg.get("instruction", "") for seg in t.get("segments", [])]
        except (KeyError, IndexError, ValueError):
            pass
        return route_info

    def _amap_distance(self, params: dict) -> ToolResult:
        origins = params.get("origins", "")
        destination = params.get("destination", "")
        if not origins or not destination:
            return ToolResult(False, error="起点或终点参数为空")
        url = f"{AMAP_BASE_URL}/distance"
        req_params = {"origins": origins, "destination": destination, "type": "1"}
        result = self._amap_request(url, req_params)
        if result.get("status") == "1" and result.get("results"):
            r = result["results"][0]
            return ToolResult(True, data={
                "distance": f"{float(r.get('distance', 0))/1000:.1f}公里",
                "duration": f"{int(r.get('duration', 0))//60}分钟",
            })
        info = result.get("info", "未知错误")
        return ToolResult(False, error=f"距离测量失败：{info}")

    def _amap_weather(self, params: dict) -> ToolResult:
        city = params.get("city", "")
        extensions = params.get("extensions", "base")
        if not city:
            return ToolResult(False, error="城市参数为空")
        geo = self._amap_geocode({"address": city})
        if not geo.success:
            return ToolResult(False, error=f"无法解析城市「{city}」：{geo.error}")
        adcode = geo.data.get("adcode", "")
        if not adcode:
            return ToolResult(False, error=f"无法获取城市「{city}」的区域编码")
        url = f"{AMAP_BASE_URL}/weather/weatherInfo"
        result = self._amap_request(url, {"city": adcode, "extensions": extensions})
        if result.get("status") != "1":
            info = result.get("info", "未知错误")
            return ToolResult(False, error=f"天气查询失败：{info}")
        if extensions == "all" and result.get("forecasts"):
            fc = result["forecasts"][0]
            return ToolResult(True, data={
                "type": "forecast",
                "city": fc.get("city", city),
                "province": fc.get("province", ""),
                "reporttime": fc.get("reporttime", ""),
                "casts": fc.get("casts", []),
            })
        if result.get("lives"):
            live = result["lives"][0]
            return ToolResult(True, data={
                "type": "live",
                "city": live.get("city", city),
                "province": live.get("province", ""),
                "weather": live.get("weather", ""),
                "temperature": live.get("temperature", ""),
                "winddirection": live.get("winddirection", ""),
                "windpower": live.get("windpower", ""),
                "humidity": live.get("humidity", ""),
                "reporttime": live.get("reporttime", ""),
            })
        return ToolResult(False, error=f"未获取到天气数据（城市：{city}）")


# ==========================================================================
# 六、Agent 生态 (Agent Ecosystem)
# ==========================================================================

class BaseAgent:
    """Agent 基类 — spec.md §5.3"""

    def __init__(self, agent_id: str, name: str, domain: ServiceDomain,
                 version: str = "1.0.0", permission_level: PermissionLevel = PermissionLevel.MEDIUM):
        self.agent_id = agent_id
        self.name = name
        self.domain = domain
        self.version = version
        self.permission_level = permission_level
        self.status = AgentStatus.ENABLED
        self.allowed_tools: list = []
        self.capabilities: list = []
        self._call_count = 0
        self._success_count = 0

    @property
    def descriptor(self) -> AgentDescriptor:
        return AgentDescriptor(
            agent_id=self.agent_id, name=self.name, domain=self.domain,
            version=self.version, capabilities=self.capabilities,
            allowed_tools=self.allowed_tools, permission_level=self.permission_level,
            status=self.status
        )

    def execute(self, intent: Intent, context: SessionContext,
                shared_context: dict, tools: ToolExecutor) -> SubTaskResult:
        self._call_count += 1
        sub_task_id = str(uuid.uuid4())[:8]
        try:
            response, data = self._process(intent, context, shared_context, tools)
            self._success_count += 1
            return SubTaskResult(sub_task_id, self.agent_id, True, response, data)
        except Exception as e:
            return SubTaskResult(sub_task_id, self.agent_id, False,
                                 f"执行失败: {str(e)}", error=str(e))

    def _process(self, intent: Intent, context: SessionContext,
                 shared_context: dict, tools: ToolExecutor) -> tuple:
        raise NotImplementedError

    @property
    def stats(self) -> dict:
        rate = self._success_count / self._call_count * 100 if self._call_count > 0 else 0
        return {"calls": self._call_count, "success": self._success_count, "rate": f"{rate:.1f}%"}


class ScheduleStore:
    """日程本地存储 — JSON 文件持久化"""

    _data: dict = {}
    _file_path: str = ""
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls):
        if cls._loaded:
            return
        cls._file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "schedules.json")
        if os.path.exists(cls._file_path):
            try:
                with open(cls._file_path, "r", encoding="utf-8") as f:
                    cls._data = json.load(f)
            except Exception:
                cls._data = {}
        cls._loaded = True

    @classmethod
    def _save(cls):
        try:
            with open(cls._file_path, "w", encoding="utf-8") as f:
                json.dump(cls._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def add(cls, date: str, time_str: str, event: str) -> dict:
        cls._ensure_loaded()
        if date not in cls._data:
            cls._data[date] = []
        entry = {"time": time_str, "event": event}
        cls._data[date].append(entry)
        cls._save()
        return entry

    @classmethod
    def query(cls, date: str) -> list:
        cls._ensure_loaded()
        return cls._data.get(date, [])

    _CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}

    @classmethod
    def _cn_to_int(cls, s: str) -> int:
        if not s:
            return 0
        if s == "十":
            return 10
        if "十" in s:
            parts = s.split("十")
            tens = cls._CN_DIGITS.get(parts[0], 1) if parts[0] else 1
            ones = cls._CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return tens * 10 + ones
        return cls._CN_DIGITS.get(s, 0)

    @staticmethod
    def resolve_date(date_str: str) -> str:
        today = datetime.now()
        if date_str == "今天":
            return today.strftime("%Y-%m-%d")
        if date_str == "明天":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if date_str == "后天":
            return (today + timedelta(days=2)).strftime("%Y-%m-%d")
        m = re.match(r'(\d{1,2})月(\d{1,2})日', date_str)
        if m:
            return f"{today.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
        if m:
            return date_str
        return date_str


class ScheduleAgent(BaseAgent):
    """日程管理 Agent — spec.md §5.3.1.1"""

    _ADD_KEYWORDS = ["安排", "记录", "添加", "创建", "设置提醒", "帮我记"]
    _QUERY_KEYWORDS = ["查询", "查看", "有什么安排", "什么安排", "哪些安排", "有哪些"]
    _CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}

    def __init__(self):
        super().__init__("agent_schedule", "日程管理Agent", ServiceDomain.SCHEDULE)
        self.allowed_tools = ["calendar_query", "calendar_add"]
        self.capabilities = ["创建日程", "查询日程", "修改日程", "删除日程", "设置提醒"]

    @classmethod
    def _cn_to_int(cls, s: str) -> int:
        if not s:
            return 0
        if s == "十":
            return 10
        if "十" in s:
            parts = s.split("十")
            tens = cls._CN_DIGITS.get(parts[0], 1) if parts[0] else 1
            ones = cls._CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return tens * 10 + ones
        return cls._CN_DIGITS.get(s, 0)

    @classmethod
    def _extract_date(cls, query: str) -> str:
        if "后天" in query:
            return "后天"
        if "明天" in query:
            return "明天"
        if "今天" in query:
            return "今天"
        m = re.search(r'(\d{1,2})月(\d{1,2})(?:日|号)', query)
        if m:
            return f"{int(m.group(1))}月{int(m.group(2))}日"
        m = re.search(r'([一二三四五六七八九十]+)月([一二三四五六七八九十]+)(?:日|号)', query)
        if m:
            month = cls._cn_to_int(m.group(1))
            day = cls._cn_to_int(m.group(2))
            if month and day:
                return f"{month}月{day}日"
        m = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', query)
        if m:
            return m.group(1)
        return "今天"

    @classmethod
    def _extract_time(cls, query: str) -> str:
        m = re.search(
            r'((?:上午|下午|晚上|早上)?\s*'
            r'(?:\d{1,2}|[一二三四五六七八九十]+)[点时]'
            r'(?:(?:\d{1,2}|[一二三四五六七八九十]+)分)?)', query)
        return m.group(1).strip() if m else ""

    @classmethod
    def _normalize_time(cls, time_str: str) -> str:
        if not time_str:
            return ""
        period = ""
        for p in ["上午", "下午", "晚上", "早上"]:
            if p in time_str:
                period = p
                break
        hour = 0
        minute = 0
        m = re.search(r'(\d{1,2})[点时]', time_str)
        if m:
            hour = int(m.group(1))
        else:
            m = re.search(r'([一二三四五六七八九十]+)[点时]', time_str)
            if m:
                hour = cls._cn_to_int(m.group(1))
        m = re.search(r'[点时](\d{1,2})分', time_str)
        if m:
            minute = int(m.group(1))
        else:
            m = re.search(r'[点时]([一二三四五六七八九十]+)分', time_str)
            if m:
                minute = cls._cn_to_int(m.group(1))
        if period in ("下午", "晚上") and hour < 12:
            hour += 12
        return f"{hour:02d}:{minute:02d}"

    @classmethod
    def _extract_event(cls, query: str) -> str:
        event = re.sub(
            r'(给我|帮我|请|安排|记录|添加|创建|设置|提醒|明天|后天|今天|'
            r'下午|上午|晚上|早上|什么|有|哪些|的|查询|查看|一下|下|'
            r'\d{1,2}月\d{1,2}(?:日|号)|'
            r'[一二三四五六七八九十]+月[一二三四五六七八九十]+(?:日|号)|'
            r'\d{4}-\d{1,2}-\d{1,2}|'
            r'(?:上午|下午|晚上|早上)?\s*\d{1,2}[点时](?:\d{1,2}分)?|'
            r'(?:上午|下午|晚上|早上)?\s*[一二三四五六七八九十]+[点时](?:[一二三四五六七八九十]+分)?|'
            r'\s)',
            '', query).strip()
        return event or "新事件"

    def _process(self, intent, context, shared_context, tools):
        query = intent.query
        is_add = any(k in query for k in self._ADD_KEYWORDS)
        is_query = any(k in query for k in self._QUERY_KEYWORDS)

        if is_add and not is_query:
            return self._add_schedule(query, tools)
        if is_query or ("安排" in query and "什么" in query):
            return self._query_schedule(query, tools)

        response = "📅 我是日程管理助手，可以帮您查询、创建、修改日程和设置提醒。请问您需要什么帮助？"
        return response, {}

    def _add_schedule(self, query: str, tools) -> tuple:
        date = self._extract_date(query)
        time_str = self._extract_time(query)
        time_norm = self._normalize_time(time_str)
        event = self._extract_event(query)
        result = tools.invoke(self.agent_id, "calendar_add",
                              {"date": date, "time": time_norm, "event": event})
        if not result.success:
            return f"❌ 创建日程失败：{result.error}", {}
        resolved = result.data.get("resolved_date", date)
        time_display = f" {time_norm}" if time_norm else ""
        response = (f"✅ 已为您创建日程：{event}（【{resolved}】{time_display}）\n"
                    f"  将在开始前15分钟提醒您")
        return response, result.data

    def _query_schedule(self, query: str, tools) -> tuple:
        date = self._extract_date(query)
        result = tools.invoke(self.agent_id, "calendar_query", {"date": date})
        if not result.success:
            return f"❌ 查询日程失败：{result.error}", {}
        events = result.data.get("events", [])
        resolved = result.data.get("resolved_date", date)
        if not events:
            return f"📅 【{resolved}】暂无安排。", {"events": [], "date": date}
        sorted_events = sorted(events, key=lambda e: e.get("time", "99:99"))
        response = f"📅 【{resolved}】的日程安排：\n"
        for e in sorted_events:
            time_val = e.get("time", "")
            time_display = f"{time_val} - " if time_val else ""
            response += f"  • {time_display}{e.get('event', '')}\n"
        return response, {"events": sorted_events, "date": date}


class HealthAgent(BaseAgent):
    """健康顾问 Agent — spec.md §5.3.1.1"""

    _HEALTH_KB = {
        "cold": {
            "keywords": ["感冒", "发烧", "咳嗽", "流鼻涕", "嗓子疼", "喉咙痛", "鼻塞", "打喷嚏"],
            "title": "🤧 感冒/呼吸道建议",
            "advice": [
                "多喝温水，保持每日2000ml以上饮水量，有助于缓解喉咙不适和稀释痰液",
                "保证充足休息，成人每日7-8小时睡眠，避免熬夜和过度劳累",
                "饮食清淡，多吃富含维C的水果（橙子、猕猴桃）和蔬菜，避免辛辣油腻",
                "保持室内通风，湿度40%-60%为宜，可使用加湿器缓解鼻塞",
                "如体温超过38.5°C或症状持续3天以上，请及时就医",
            ],
        },
        "fitness": {
            "keywords": ["健身", "锻炼", "运动", "跑步", "增肌", "有氧", "无氧", "拉伸"],
            "title": "🏃 运动/健身建议",
            "advice": [
                "成人每周至少150分钟中等强度有氧运动，或75分钟高强度运动",
                "运动前热身5-10分钟，运动后拉伸放松，降低受伤风险",
                "增肌训练建议每组8-12次，组间休息60-90秒，每周每个肌群训练2次",
                "运动时心率控制在最大心率（220-年龄）的60%-80%区间",
                "避免空腹高强度运动，运动前1小时可补充少量碳水",
            ],
        },
        "diet": {
            "keywords": ["饮食", "吃", "营养", "食谱", "吃什么", "膳食", "卡路里"],
            "title": "🥗 饮食/营养建议",
            "advice": [
                "每日均衡摄入：谷物50-65%、蔬菜水果30-35%、蛋白质15-20%",
                "早餐要吃好（全麦+鸡蛋+牛奶），午餐要吃饱，晚餐要吃少",
                "每日盐摄入<5g，油25-30g，添加糖<25g，避免加工食品",
                "多选优质蛋白：鱼、鸡胸肉、豆腐、鸡蛋，减少红肉和加工肉",
                "每日饮水1500-1700ml，少喝含糖饮料，可用白开水或淡茶替代",
            ],
        },
        "sleep": {
            "keywords": ["睡眠", "失眠", "休息", "睡不着", "入睡", "熬夜", "做梦"],
            "title": "😴 睡眠/休息建议",
            "advice": [
                "成人每日保持7-9小时睡眠，固定作息时间，周末不宜偏差超过1小时",
                "睡前1小时避免使用手机/电脑，蓝光会抑制褪黑素分泌",
                "卧室温度保持18-22°C，保持黑暗安静，可使用遮光窗帘和耳塞",
                "睡前避免咖啡因（下午2点后不喝咖啡/茶）和酒精",
                "如长期失眠（>3次/周，持续>1月），建议就医排查睡眠障碍",
            ],
        },
        "weight_loss": {
            "keywords": ["减肥", "瘦身", "减脂", "减重", "卡路里", "热量", "BMI"],
            "title": "⚖️ 减肥/体重管理建议",
            "advice": [
                "科学减重速度：每周0.5-1kg，通过每日热量缺口300-500kcal实现",
                "饮食控制为主（70%），运动为辅（30%），避免极端节食",
                "增加蛋白质比例（1.2-1.6g/kg体重），减少精制碳水，增加膳食纤维",
                "推荐运动：慢跑、游泳、HIIT，每周3-5次，每次30-60分钟",
                "记录每日饮食和体重变化，定期评估调整方案",
            ],
        },
        "headache": {
            "keywords": ["头痛", "头疼", "头晕", "偏头痛", "眩晕"],
            "title": "🤕 头痛/头晕建议",
            "advice": [
                "常见原因：紧张性头痛（压力/姿势）、偏头痛（血管）、颈源性头痛",
                "立即缓解：在安静暗处休息，太阳穴按摩，冷敷额头10-15分钟",
                "注意补充水分，低血糖和脱水是常见诱因",
                "避免长时间低头看手机，每30分钟活动颈部和肩部",
                "如头痛剧烈伴呕吐/视力模糊/肢体麻木，请立即就医",
            ],
        },
        "stomach": {
            "keywords": ["胃痛", "肠胃", "消化", "拉肚子", "腹泻", "便秘", "胃酸", "胀气"],
            "title": "🫁 肠胃/消化建议",
            "advice": [
                "规律饮食，三餐定时定量，避免暴饮暴食和睡前进食",
                "腹泻时注意补充水分和电解质，可喝淡盐水或口服补液盐",
                "便秘时增加膳食纤维（蔬菜、全谷物、火龙果），每日饮水>2000ml",
                "胃酸过多避免空腹喝咖啡/浓茶，可少量多餐，饭后不要立即躺下",
                "如症状持续超过1周或伴便血/剧烈腹痛，请及时就医",
            ],
        },
        "mental": {
            "keywords": ["压力", "焦虑", "抑郁", "心理", "情绪", "烦躁", "紧张"],
            "title": "🧠 心理/情绪健康建议",
            "advice": [
                "每日10-15分钟正念冥想或深呼吸练习，有效缓解焦虑",
                "保持社交联系，与亲友定期交流，避免长期独处",
                "规律运动是天然抗抑郁剂，每周3次以上中等强度运动",
                "学会区分可控/不可控因素，专注当下可控之事",
                "如情绪低落持续>2周或影响正常生活，建议寻求专业心理咨询",
            ],
        },
        "water": {
            "keywords": ["喝水", "饮水", "补水", "脱水", "口渴", "喝多少水", "喝"],
            "title": "💧 饮水/补水建议",
            "advice": [
                "成人每日饮水量1500-1700ml（约8杯），运动后额外补充",
                "不要等口渴才喝水，口渴时身体已轻度脱水",
                "最佳饮水时间：晨起1杯、餐前30分钟、下午茶时间、睡前少量",
                "运动补水：每15-20分钟补充150-200ml，避免一次性大量饮用",
                "观察尿液颜色，淡黄色为正常，深黄色提示需要补水",
            ],
        },
        "data": {
            "keywords": ["步数", "走了", "心率", "心跳", "数据", "健康数据"],
            "title": "📊 今日健康数据",
            "advice": [],
        },
    }

    _HEALTH_SYSTEM_PROMPT = (
        "你是一位专业的健康顾问AI助手。请根据用户的具体健康问题，提供准确、实用、安全的健康建议。\n"
        "要求：\n"
        "1. 用简体中文回答，条理清晰，使用编号列表格式\n"
        "2. 给出具体的、可操作的建议，避免空泛的套话\n"
        "3. 如涉及严重症状或紧急情况，明确提示用户及时就医\n"
        "4. 回答控制在300字以内，简洁有力\n"
        "5. 末尾添加免责声明：以上建议仅供参考，不能替代专业医疗诊断"
    )

    def __init__(self):
        super().__init__("agent_health", "健康顾问Agent", ServiceDomain.HEALTH)
        self.allowed_tools = ["health_query"]
        self.capabilities = ["健康数据查询", "运动建议", "饮食建议", "健康提醒"]

    _TOPIC_PRIORITY = ["cold", "headache", "stomach", "sleep", "mental",
                        "weight_loss", "fitness", "water", "diet", "data"]
    _TOPIC_WEIGHT = {"cold": 3, "headache": 3, "stomach": 3,
                     "sleep": 2, "mental": 2, "weight_loss": 2,
                     "fitness": 2, "water": 2, "diet": 1, "data": 1}

    @classmethod
    def _match_topic(cls, query: str) -> str:
        scored = []
        for topic, info in cls._HEALTH_KB.items():
            score = sum(1 for k in info["keywords"] if k in query)
            if score > 0:
                weight = cls._TOPIC_WEIGHT.get(topic, 1)
                priority = len(cls._TOPIC_PRIORITY) - cls._TOPIC_PRIORITY.index(topic)
                scored.append((score * weight * 100 + priority, topic))
        if not scored:
            return "general"
        scored.sort(reverse=True)
        return scored[0][1]

    def _process(self, intent, context, shared_context, tools):
        query = intent.query
        topic = self._match_topic(query)
        result = tools.invoke(self.agent_id, "health_query", {"topic": topic})
        data = result.data

        if topic == "data":
            response = (f"{self._HEALTH_KB['data']['title']}：\n"
                        f"  • 步数: {data['steps']} 步\n"
                        f"  • 心率: {data['heart_rate']} bpm\n"
                        f"  • 睡眠: {data['sleep']}")
            if data['steps'] < 10000:
                response += f"\n  💡 建议再走 {10000 - data['steps']} 步达到每日目标"
            return response, data

        llm_response = _call_llm(query, self._HEALTH_SYSTEM_PROMPT)
        if llm_response:
            return f"💊 {llm_response}", {"topic": topic, "source": "llm",
                                           "response": llm_response}

        if topic in self._HEALTH_KB:
            info = self._HEALTH_KB[topic]
            response = f"{info['title']}：\n"
            for i, tip in enumerate(info["advice"], 1):
                response += f"  {i}. {tip}\n"
            return response, {"topic": topic, "source": "knowledge_base",
                              "advice": info["advice"]}

        response = ("💊 我是健康顾问，可以为您提供以下方面的健康建议：\n"
                    "  • 感冒/发烧/咳嗽等常见疾病注意事项\n"
                    "  • 运动/健身/锻炼指导\n"
                    "  • 饮食/营养/膳食建议\n"
                    "  • 睡眠/失眠改善方案\n"
                    "  • 减肥/体重管理\n"
                    "  • 心理/情绪健康\n"
                    "  • 头痛/肠胃等不适缓解\n"
                    "  💡 请告诉我您具体的健康问题，我将为您提供针对性建议。")
        return response, {"topic": "general", "source": "fallback"}


class VirtualDeviceState:
    """虚拟设备状态管理 — Mock State for smart home simulation"""

    _DEVICE_STATE = {
        "客厅灯": {"power": False, "brightness": 80, "icon": "💡"},
        "卧室灯": {"power": False, "brightness": 60, "icon": "💡"},
        "空调": {"power": False, "temperature": 26, "mode": "制冷", "icon": "❄️"},
        "电视": {"power": False, "volume": 30, "icon": "📺"},
        "窗帘": {"power": True, "position": 50, "icon": "🪟"},
        "热水器": {"power": False, "temperature": 45, "icon": "🔥"},
        "扫地机器人": {"power": False, "status": "待机", "icon": "🤖"},
    }

    _SCENES = {
        "回家": {"客厅灯": {"power": True, "brightness": 80},
                 "空调": {"power": True, "temperature": 26},
                 "窗帘": {"power": True, "position": 100}},
        "离家": {"客厅灯": {"power": False}, "卧室灯": {"power": False},
                 "空调": {"power": False}, "电视": {"power": False},
                 "窗帘": {"power": False, "position": 0},
                 "热水器": {"power": False}, "扫地机器人": {"power": False}},
        "睡眠": {"客厅灯": {"power": False}, "卧室灯": {"power": False},
                 "空调": {"power": True, "temperature": 25, "mode": "制冷"},
                 "窗帘": {"power": False, "position": 0},
                 "电视": {"power": False}},
    }

    @classmethod
    def get_state(cls) -> dict:
        return cls._DEVICE_STATE

    @classmethod
    def apply_command(cls, command: dict) -> str:
        device = command.get("device", "")
        action = command.get("action", "")
        params = command.get("params", {})
        state = cls.get_state()
        if device not in state:
            return f"⚠️ 未找到设备「{device}」"
        dev = state[device]
        if action == "open":
            dev["power"] = True
            dev.update(params)
        elif action == "close":
            dev["power"] = False
        elif action == "set":
            dev.update(params)
            if not dev.get("power") and any(
                k in params for k in ("temperature", "brightness", "position", "volume")
            ):
                dev["power"] = True
        return ""

    @classmethod
    def apply_scene(cls, scene_name: str) -> list:
        scene = cls._SCENES.get(scene_name)
        if not scene:
            return []
        state = cls.get_state()
        changed = []
        for device, attrs in scene.items():
            if device in state:
                state[device].update(attrs)
                changed.append(device)
        return changed

    @classmethod
    def format_status(cls) -> str:
        state = cls.get_state()
        lines = []
        for device, info in state.items():
            icon = info.get("icon", "🔹")
            if info.get("power"):
                if "temperature" in info:
                    lines.append(f"  {icon} {device}: ✅开启 {info['temperature']}°C {info.get('mode', '')}")
                elif "brightness" in info:
                    lines.append(f"  {icon} {device}: ✅开启 亮度{info['brightness']}%")
                elif "position" in info:
                    lines.append(f"  {icon} {device}: ✅开启 {info['position']}%")
                elif "volume" in info:
                    lines.append(f"  {icon} {device}: ✅开启 音量{info['volume']}")
                elif "status" in info:
                    lines.append(f"  {icon} {device}: ✅{info['status']}")
                else:
                    lines.append(f"  {icon} {device}: ✅开启")
            else:
                lines.append(f"  {icon} {device}: ❌关闭")
        return "\n".join(lines)


class HomeAgent(BaseAgent):
    """智能家居控制 Agent — spec.md §5.3.1.1"""

    _HOME_SYSTEM_PROMPT = (
        "你是一位智能家居控制助手。根据用户的自然语言指令和当前设备状态，"
        "解析出控制命令并给出人性化的反馈。\n\n"
        "当前设备状态：\n{device_state}\n\n"
        "可用设备：客厅灯、卧室灯、空调、电视、窗帘、热水器、扫地机器人\n\n"
        "请严格返回以下JSON格式（不要包含```标记或其他多余文字）：\n"
        '{{"commands": [{{"device": "设备名", "action": "open|close|set|query", "params": {{}}}}], '
        '"response": "人性化的操作反馈"}}\n\n'
        "规则：\n"
        "- action: open=打开, close=关闭, set=设置参数, query=查询状态\n"
        "- 空调params: temperature(16-30), mode(制冷/制热/自动/送风)\n"
        "- 灯params: brightness(0-100)\n"
        "- 窗帘params: position(0-100, 0=全关, 100=全开)\n"
        "- 电视params: volume(0-100)\n"
        "- 热水器params: temperature(30-75)\n"
        "- 如用户查询设备状态，action=query，response中包含完整状态信息\n"
        "- 如用户说场景模式（回家/离家/睡眠），返回多个commands批量控制\n"
        "- response用简体中文，简洁友好，包含操作结果"
    )

    def __init__(self):
        super().__init__("agent_home", "智能家居Agent", ServiceDomain.HOME,
                         permission_level=PermissionLevel.HIGH)
        self.allowed_tools = ["device_control"]
        self.capabilities = ["设备控制", "设备状态查询", "场景模式", "定时任务"]

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    def _process(self, intent, context, shared_context, tools):
        query = intent.query
        state_str = VirtualDeviceState.format_status()
        system_prompt = self._HOME_SYSTEM_PROMPT.format(device_state=state_str)

        llm_response = _call_llm(query, system_prompt)
        if llm_response:
            result = self._parse_and_execute(llm_response)
            if result:
                return result

        response = ("🏠 抱歉，智能家居控制服务暂不可用。\n"
                    "  ⚠️ AI 引擎调用失败，请检查 DeepSeek API 配置和余额。\n"
                    "  💡 您可以稍后重试。")
        return response, {"source": "llm_error"}

    def _parse_and_execute(self, llm_response: str):
        data = self._extract_json(llm_response)
        if not data:
            return None

        commands = data.get("commands", [])
        response_text = data.get("response", "")
        if not commands and not response_text:
            return None

        executed = []
        for cmd in commands:
            action = cmd.get("action", "")
            device = cmd.get("device", "")
            params = cmd.get("params", {})

            if action in ("open", "close", "set"):
                VirtualDeviceState.apply_command(cmd)
                executed.append(f"{action}:{device}")
            elif action == "query":
                executed.append(f"query:{device}")

        status = VirtualDeviceState.format_status()
        full_response = f"🏠 {response_text}\n\n📋 当前设备状态：\n{status}"
        return full_response, {"source": "llm", "executed": executed,
                               "device_state": VirtualDeviceState.get_state()}


class ShoppingAgent(BaseAgent):
    """购物助手 Agent — spec.md §5.3.1.1"""

    _SHOPPING_KB = {
        "recommend": {
            "title": "🛒 好物推荐",
            "tips": [
                "智能水杯 ¥89 — 提醒喝水好帮手，APP联动记录饮水量",
                "颈椎按摩仪 ¥299 — 办公族必备，缓解久坐疲劳",
                "空气净化器 ¥599 — 呼吸健康保障，PM2.5去除率99%",
                "电动牙刷 ¥159 — 声波震动清洁，2分钟定时关机",
                "蓝牙耳机 ¥199 — 降噪通话清晰，续航20小时",
            ],
        },
        "compare": {
            "title": "🛒 比价参考",
            "tips": [
                "京东：正品保障，PLUS会员额外折扣，物流次日达",
                "淘宝：品类丰富，优惠券多，天猫旗舰店品质有保障",
                "拼多多：百亿补贴价格最低，注意辨别正品",
                "比价建议：综合价格+运费+售后，不唯低价论",
            ],
        },
        "list": {
            "title": "🛒 购物清单",
            "tips": [
                "日用刚需：纸巾、洗衣液、垃圾袋、洗洁精",
                "生鲜食品：鸡蛋、牛奶、蔬菜、水果",
                "个人护理：牙膏、洗发水、沐浴露、面膜",
                "零食饮料：坚果、酸奶、咖啡、茶包",
            ],
        },
    }

    _RECOMMEND_PROMPT = (
        "你是一位专业的购物推荐顾问。请根据用户的需求，推荐3-5款值得购买的商品。\n"
        "要求：\n"
        "1. 用简体中文回答，格式为编号列表\n"
        "2. 每款商品包含：名称、参考价格、推荐理由（一句话）\n"
        "3. 推荐应覆盖不同价位段，满足不同预算需求\n"
        "4. 优先推荐口碑好、性价比高的产品\n"
        "5. 回答控制在300字以内"
    )

    _COMPARE_PROMPT = (
        "你是一位专业的比价顾问。请根据用户的需求，从多个维度进行价格和参数比较。\n"
        "要求：\n"
        "1. 用简体中文回答，条理清晰，使用编号列表\n"
        "2. 比较维度包括：价格、品质、售后、物流等\n"
        "3. 给出明确的购买建议，说明推荐理由\n"
        "4. 回答控制在300字以内"
    )

    _LIST_PROMPT = (
        "你是一位专业的购物清单规划师。请根据用户的需求，生成一份实用的购物清单。\n"
        "要求：\n"
        "1. 用简体中文回答，按类别分组\n"
        "2. 每类列出2-4项具体商品\n"
        "3. 如用户指定了场景（如搬家、囤货、日常等），按场景调整清单\n"
        "4. 回答控制在300字以内"
    )

    _SEARCH_PROMPT = (
        "你是一位专业的购物搜索助手。请根据用户的需求，推荐相关商品并给出购买建议。\n"
        "要求：\n"
        "1. 用简体中文回答，格式为编号列表\n"
        "2. 每款商品包含：名称、参考价格、简要说明\n"
        "3. 回答控制在300字以内"
    )

    def __init__(self):
        super().__init__("agent_shopping", "购物助手Agent", ServiceDomain.SHOPPING)
        self.allowed_tools = ["product_search"]
        self.capabilities = ["商品搜索", "价格比较", "购物推荐", "购物清单", "订单查询"]

    @classmethod
    def _classify_request(cls, query: str) -> str:
        if any(k in query for k in ["清单", "列个单", "买些什么", "买点啥",
                                     "采购", "囤货", "补货"]):
            return "list"
        if any(k in query for k in ["推荐", "好物", "种草", "买什么",
                                     "值得买", "排行", "榜单", "比较好"]):
            return "recommend"
        if any(k in query for k in ["比价", "比较", "哪个好", "性价比",
                                     "划算", "便宜", "贵", "评测", "测评",
                                     "优惠券", "满减", "折扣", "促销"]):
            return "compare"
        if any(k in query for k in ["搜索", "查找", "找", "买", "购买",
                                     "下单", "选购", "挑选"]):
            return "search"
        return "recommend"

    @classmethod
    def _extract_product_keyword(cls, query: str) -> str:
        keyword = re.sub(
            r'(帮我|请|搜索|查找|买|购买|找|推荐|想|要|一些|个|件|什么|好物|种草|比价|比较|哪个好|性价比|划算|清单|列个单|买些什么|买点啥|采购|便宜|贵|优惠券|促销|满减|折扣|评测|测评|排行|榜单|值得买|囤货|补货|选购|挑选|一下|看看|帮我|能否|可以|我|一|两|三|四|五|六|七|八|九|十|台|款|双|条|把|支|瓶|包|箱|套|组|对|副|张|块|部|辆|顶|件|双)\s*',
            '', query).strip()
        return keyword

    def _process(self, intent, context, shared_context, tools):
        query = intent.query
        req_type = self._classify_request(query)
        keyword = self._extract_product_keyword(query)

        prompt_map = {
            "recommend": self._RECOMMEND_PROMPT,
            "compare": self._COMPARE_PROMPT,
            "list": self._LIST_PROMPT,
            "search": self._SEARCH_PROMPT,
        }

        llm_response = _call_llm(query, prompt_map[req_type])
        if llm_response:
            return f"🛒 {llm_response}", {"type": req_type, "source": "llm",
                                           "keyword": keyword,
                                           "response": llm_response}

        type_labels = {"recommend": "商品推荐", "compare": "参数比价",
                       "list": "购物清单", "search": "商品搜索"}
        response = (f"🛒 抱歉，{type_labels.get(req_type, '购物咨询')}服务暂不可用。\n"
                    f"  ⚠️ AI 引擎调用失败，请检查 DeepSeek API 配置和余额。\n"
                    f"  💡 您可以稍后重试，或换一种方式描述您的需求。")
        return response, {"type": req_type, "source": "llm_error",
                          "keyword": keyword}


class WeatherAgent(BaseAgent):
    """天气预报 Agent — 高德天气 API + DeepSeek LLM 人性化总结"""

    _CITY_PREFIXES = ["查一下", "看一下", "看看", "查询", "帮我看", "帮我查",
                      "我想看", "我想查", "帮我", "我想", "请问", "请", "帮",
                      "查", "看", "一下"]
    _CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}

    _WEATHER_LLM_PROMPT = (
        "你是一位贴心的天气预报助手。请根据以下天气数据，用口语化、人性化的方式为用户整理天气预报，"
        "并给出实用的出行建议。\n\n"
        "要求：\n"
        "1. 用简体中文回答，语气亲切自然\n"
        "2. 包含：天气概况、温度范围、出行建议（如是否带伞、穿衣建议、防晒提醒等）\n"
        "3. 如有降雨概率，明确提醒带伞；如气温较低，建议添衣；如气温较高，注意防暑\n"
        "4. 回答控制在200字以内"
    )

    def __init__(self):
        super().__init__("agent_weather", "天气预报Agent", ServiceDomain.SEARCH)
        self.allowed_tools = ["weather_query"]
        self.capabilities = ["实时天气查询", "天气预报", "出行建议", "穿衣指数"]

    @classmethod
    def _extract_city_name(cls, query: str) -> str:
        cleaned = re.sub(
            r'(\d{1,2}月\d{1,2}(?:日|号)|'
            r'[一二三四五六七八九十]+月[一二三四五六七八九十]+(?:日|号)|'
            r'\d{4}-\d{1,2}-\d{1,2}|'
            r'今天|明天|后天|大后天|未来|几天|预报)',
            '', query)
        for keyword in ["天气", "气温", "温度", "下雨", "预报"]:
            pattern = rf'([\u4e00-\u9fa5]{{2,10}})(?:的)?{keyword}'
            match = re.search(pattern, cleaned)
            if match:
                city = match.group(1).rstrip("的")
                changed = True
                while changed:
                    changed = False
                    for p in cls._CITY_PREFIXES:
                        if city.startswith(p) and len(city) > len(p) + 1:
                            city = city[len(p):]
                            changed = True
                return city
        return "北京"

    @classmethod
    def _cn_to_int(cls, s: str) -> int:
        if not s:
            return 0
        if s == "十":
            return 10
        if "十" in s:
            parts = s.split("十")
            tens = cls._CN_DIGITS.get(parts[0], 1) if parts[0] else 1
            ones = cls._CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return tens * 10 + ones
        return cls._CN_DIGITS.get(s, 0)

    @classmethod
    def _resolve_weather_date(cls, query: str) -> str:
        today = datetime.now()
        if "大后天" in query:
            return (today + timedelta(days=3)).strftime("%Y-%m-%d")
        if "后天" in query:
            return (today + timedelta(days=2)).strftime("%Y-%m-%d")
        if "明天" in query:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if "今天" in query:
            return today.strftime("%Y-%m-%d")
        m = re.search(r'(\d{1,2})月(\d{1,2})(?:日|号)', query)
        if m:
            return f"{today.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        m = re.search(r'([一二三四五六七八九十]+)月([一二三四五六七八九十]+)(?:日|号)', query)
        if m:
            month = cls._cn_to_int(m.group(1))
            day = cls._cn_to_int(m.group(2))
            if month and day:
                return f"{today.year}-{month:02d}-{day:02d}"
        m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', query)
        if m:
            return m.group(0)
        return ""

    @classmethod
    def _has_explicit_date(cls, query: str) -> bool:
        return any(k in query for k in ["今天", "明天", "后天", "大后天"]) or \
               bool(re.search(r'\d{1,2}月\d{1,2}(?:日|号)', query)) or \
               bool(re.search(r'[一二三四五六七八九十]+月[一二三四五六七八九十]+(?:日|号)', query)) or \
               bool(re.search(r'\d{4}-\d{1,2}-\d{1,2}', query))

    def _process(self, intent, context, shared_context, tools):
        query = intent.query
        city = self._extract_city_name(query)
        target_date = self._resolve_weather_date(query)
        has_date = self._has_explicit_date(query)
        want_list = any(k in query for k in ["未来", "几天", "预报"]) and not has_date
        extensions = "all" if (has_date or want_list) else "base"

        result = tools.invoke(self.agent_id, "weather_query",
                              {"city": city, "extensions": extensions})
        if not result.success:
            return f"🌤️ 天气查询失败：{result.error}", {"source": "api_error"}

        d = result.data
        weather_summary = self._build_weather_text(query, d, city, has_date, target_date, want_list)

        llm_input = f"用户问题：{query}\n\n天气数据：\n{weather_summary}"
        llm_response = _call_llm(llm_input, self._WEATHER_LLM_PROMPT)
        if llm_response:
            return f"🌤️ {llm_response}", {"source": "llm", "city": city,
                                            "weather_data": d}

        return f"🌤️ {weather_summary}", {"source": "api", "city": city,
                                          "weather_data": d}

    @staticmethod
    def _build_weather_text(query, d, city, has_date, target_date, want_list):
        if d.get("type") == "live":
            text = (f"{d['city']}实况天气（{d.get('reporttime', '')}）\n"
                    f"天气：{d.get('weather', '未知')}，气温{d.get('temperature', '未知')}°C\n"
                    f"风向：{d.get('winddirection', '未知')}，风力{d.get('windpower', '未知')}级\n"
                    f"湿度：{d.get('humidity', '未知')}%")
            weather_text = d.get("weather", "")
            if "雨" in weather_text:
                text += "\n当前有雨，建议带伞"
            elif "伞" in query:
                text += "\n当前无雨，无需带伞"
            return text

        casts = d.get("casts", [])
        if has_date and target_date:
            target_day = None
            for cast in casts:
                if cast.get("date") == target_date:
                    target_day = cast
                    break
            if target_day:
                text = (f"{d.get('city', city)}天气预报（{target_day.get('date', '')}）\n"
                        f"白天：{target_day.get('dayweather', '未知')}，{target_day.get('daytemp', '未知')}°C\n"
                        f"夜间：{target_day.get('nightweather', '未知')}，{target_day.get('nighttemp', '未知')}°C\n"
                        f"风向：{target_day.get('daywind', '未知')}，风力{target_day.get('daypower', '未知')}级")
                day_w = target_day.get("dayweather", "")
                night_w = target_day.get("nightweather", "")
                if "雨" in day_w or "雨" in night_w:
                    text += "\n有雨，建议带伞"
                return text
            date_range = f"{casts[0].get('date')}~{casts[-1].get('date')}" if casts else "无数据"
            return f"暂无{target_date}的天气预报数据。当前预报覆盖范围：{date_range}"

        text = f"{d.get('city', city)}未来天气预报（发布于{d.get('reporttime', '')}）\n"
        for cast in casts[:4]:
            text += (f"{cast.get('date', '')} 周{cast.get('week', '')}: "
                     f"白天{cast.get('dayweather', '')} {cast.get('daytemp', '')}°C / "
                     f"夜间{cast.get('nightweather', '')} {cast.get('nighttemp', '')}°C\n")
        return text


class RouteAgent(BaseAgent):
    """路线规划 Agent — 高德地图 API 集成"""

    def __init__(self):
        super().__init__("agent_route", "路线规划Agent", ServiceDomain.ROUTE)
        self.allowed_tools = ["amap_geocode", "amap_direction_driving",
                              "amap_direction_walking", "amap_direction_transit",
                              "amap_distance"]
        self.capabilities = ["驾车路线规划", "步行路线规划", "公交路线规划",
                             "地址解析", "距离测量", "出行方案推荐"]

    def _process(self, intent, context, shared_context, tools):
        query = intent.query
        slots = intent.slots
        origin = slots.get("origin", "")
        destination = slots.get("destination", "")

        if not origin and not destination:
            origin, destination = self._extract_locations(query)

        if any(k in query for k in ["步行", "走路", "徒步"]):
            return self._plan_route(tools, origin, destination, "walking")
        elif any(k in query for k in ["公交", "地铁", "坐车", "乘车", "公共交通"]):
            return self._plan_route(tools, origin, destination, "transit")
        elif any(k in query for k in ["距离", "多远", "多少公里"]):
            return self._measure_distance(tools, origin, destination)
        elif any(k in query for k in ["驾车", "开车", "自驾", "开车去", "驾车去"]) or (origin and destination):
            return self._plan_route(tools, origin, destination, "driving")
        elif any(k in query for k in ["路线", "导航", "怎么走", "怎么去", "从", "到"]):
            return self._plan_route(tools, origin, destination, "driving")

        response = "🧭 我是路线规划助手，可以帮您规划驾车、步行、公交路线，查询距离和出行方案。\n  例如：「从天安门到故宫怎么走」「步行从公司到地铁站」"
        return response, {}

    def _plan_route(self, tools, origin: str, destination: str, mode: str) -> tuple:
        if not origin or not destination:
            return ("🧭 请提供起点和终点信息。\n  例如：「从北京站到天安门怎么走」", {})

        tool_map = {
            "driving": "amap_direction_driving",
            "walking": "amap_direction_walking",
            "transit": "amap_direction_transit",
        }
        mode_names = {"driving": "驾车", "walking": "步行", "transit": "公交"}
        mode_icons = {"driving": "🚗", "walking": "🚶", "transit": "🚌"}

        tool_id = tool_map.get(mode, "amap_direction_driving")
        params = {"origin": origin, "destination": destination}
        if mode == "driving":
            params["strategy"] = "0"
        elif mode == "transit":
            params["city"] = "北京"

        result = tools.invoke(self.agent_id, tool_id, params)
        if not result.success:
            return f"🧭 路线规划失败：{result.error}", {}

        data = result.data
        icon = mode_icons.get(mode, "🧭")
        mode_name = mode_names.get(mode, mode)

        response = f"{icon} {mode_name}路线规划：{origin} → {destination}\n"
        response += f"  📏 总距离：{data.get('distance', '未知')}\n"
        response += f"  ⏱️ 预计时间：{data.get('duration', '未知')}\n"

        steps = data.get("steps", [])
        if steps:
            response += "  📋 详细路线：\n"
            for i, step in enumerate(steps[:8], 1):
                clean_step = re.sub(r'<[^>]+>', '', step).strip()
                if clean_step:
                    response += f"    {i}. {clean_step}\n"
            if len(steps) > 8:
                response += f"    ...（共{len(steps)}步）\n"

        return response, data

    def _measure_distance(self, tools, origin: str, destination: str) -> tuple:
        if not origin or not destination:
            return "🧭 请提供起点和终点来测量距离。", {}

        if "," not in origin:
            geo = tools.invoke(self.agent_id, "amap_geocode", {"address": origin})
            if geo.success:
                origin = geo.data.get("location", origin)
            else:
                return f"🧭 无法解析起点地址「{origin}」：{geo.error}", {}
        if "," not in destination:
            geo = tools.invoke(self.agent_id, "amap_geocode", {"address": destination})
            if geo.success:
                destination = geo.data.get("location", destination)
            else:
                return f"🧭 无法解析终点地址「{destination}」：{geo.error}", {}

        result = tools.invoke(self.agent_id, "amap_distance",
                              {"origins": origin, "destination": destination})
        if not result.success:
            return f"🧭 距离测量失败：{result.error}", {}

        data = result.data
        response = (f"📏 距离测量：{origin} → {destination}\n"
                    f"  直线距离：{data.get('distance', '未知')}\n"
                    f"  预计耗时：{data.get('duration', '未知')}")
        return response, data

    @staticmethod
    def _extract_locations(query: str) -> tuple:
        origin, destination = "", ""
        patterns = [
            r'从(.+?)(?:到|去|至)(.+?)(?:怎么|路线|多远|距离|需要|的|$)',
            r'(.+?)(?:到|去|至)(.+?)(?:怎么|路线|多远|距离|需要|的|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                origin = match.group(1).strip()
                destination = match.group(2).strip()
                for stop_word in ["怎么走", "怎么去", "路线", "多远", "距离", "需要多久"]:
                    destination = destination.replace(stop_word, "").strip()
                if origin and destination:
                    break
        return origin, destination


class AgentRegistry:
    """Agent 注册表与发现服务 — spec.md §5.3.1.2 / §5.3.1.3"""

    def __init__(self, audit: AuditLogger):
        self._agents: dict = {}
        self._audit = audit

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_id] = agent
        self._audit.log("system", "Agent注册", agent.name, "成功",
                        {"agent_id": agent.agent_id, "domain": agent.domain.value})

    def get(self, agent_id: str) -> BaseAgent:
        return self._agents.get(agent_id)

    def discover(self, domain: ServiceDomain,
                 user_permission: PermissionLevel = PermissionLevel.HIGH) -> list:
        return [a for a in self._agents.values()
                if a.domain.value == domain.value and a.status.value == AgentStatus.ENABLED.value
                and user_permission.value >= a.permission_level.value]

    def list_all(self) -> list:
        return list(self._agents.values())

    def enable(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = AgentStatus.ENABLED
            self._audit.log("admin", "Agent启用", agent.name, "成功")
            return True
        return False

    def disable(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = AgentStatus.DISABLED
            self._audit.log("admin", "Agent禁用", agent.name, "成功")
            return True
        return False


# ==========================================================================
# 七、意图识别 (Intent Recognition)
# ==========================================================================

class IntentRecognizer:
    """意图识别与多意图拆分 — spec.md §5.1"""

    DOMAIN_KEYWORDS = {
        ServiceDomain.SCHEDULE: ["日程", "安排", "会议", "提醒", "计划", "待办",
                                  "日历", "行程", "预约", "时间表"],
        ServiceDomain.HEALTH: [
            "健康", "步数", "运动", "饮食", "睡眠", "心率", "血压", "体重",
            "卡路里", "健身", "营养",
            "发烧", "发热", "咳嗽", "流鼻涕", "鼻塞", "嗓子疼", "喉咙痛",
            "打喷嚏", "感冒", "头疼", "头痛", "头晕", "偏头痛", "眩晕",
            "肚子疼", "腹痛", "胃疼", "胃痛", "胃酸", "拉肚子", "腹泻",
            "便秘", "胀气", "消化不良", "恶心", "呕吐", "过敏", "湿疹",
            "皮疹", "痘痘", "长痘", "口腔溃疡", "溃疡", "关节疼", "腰疼",
            "腰痛", "肩膀疼", "颈椎", "落枕", "麻木", "抽筋", "不舒服",
            "难受", "生病", "症状", "看病", "吃药", "吃药了", "身体", "买药",
            "失眠", "睡不着", "入睡", "多梦", "做梦", "熬夜", "疲劳",
            "乏力", "没精神", "犯困", "午休", "休息", "作息",
            "吃什么", "中饭", "午饭", "晚饭", "早餐", "夜宵", "加餐",
            "食谱", "膳食", "蛋白质", "维生素", "补血", "补钙", "补铁",
            "忌口", "生理期", "姨妈期", "经期", "孕期", "哺乳期",
            "锻炼", "跑步", "增肌", "减脂", "减肥", "瘦身", "减重",
            "有氧", "无氧", "拉伸", "瑜伽", "游泳", "走了", "多少步",
            "压力", "焦虑", "抑郁", "烦躁", "紧张", "情绪", "心理",
            "失落", "崩溃", "喝水", "饮水", "补水", "口渴", "脱水",
            "体检", "挂号", "血糖", "体脂",
        ],
        ServiceDomain.HOME: ["灯", "空调", "电视", "窗帘", "开灯", "关灯",
                              "打开", "关闭", "场景", "模式", "设备", "家电",
                              "热水器", "扫地"],
        ServiceDomain.SHOPPING: ["买", "购物", "商品", "价格", "推荐好物",
                                  "搜索商品", "比价", "订单", "下单", "打折", "推荐",
                                  "清单", "性价比", "哪个好", "买什么", "好物",
                                  "种草", "划算", "比较", "便宜", "贵", "优惠券",
                                  "促销", "满减", "折扣", "评测", "测评", "排行",
                                  "榜单", "值得买", "囤货", "补货", "选购", "挑选"],
        ServiceDomain.SEARCH: ["天气", "预报", "气温", "下雨", "温度", "空气质量",
                                "带伞", "穿衣", "穿什么", "穿几件", "出太阳", "刮风",
                                "雾霾", "台风", "冷暖", "几度", "热不热", "冷不冷",
                                "晴不晴", "会不会下雨", "会不会下雪", "紫外线", "湿度"],
        ServiceDomain.ROUTE: ["路线", "导航", "怎么走", "怎么去", "从", "到",
                              "驾车", "开车", "步行", "走路", "公交", "地铁",
                              "距离", "多远", "多少公里", "自驾", "徒步",
                              "出行", "地址", "位置", "在哪"],
    }

    STRONG_KEYWORDS = {
        ServiceDomain.HEALTH: {
            "感冒", "发烧", "发热", "头痛", "头疼", "肚子疼", "腹痛", "胃疼",
            "胃痛", "血糖", "血压", "心率", "过敏", "失眠", "熬夜", "抑郁",
            "焦虑", "健身", "锻炼", "减肥", "瘦身", "增肌", "减脂", "瑜伽",
            "跑步", "游泳", "拉肚子", "腹泻", "便秘", "湿疹", "溃疡",
            "颈椎", "落枕", "抽筋", "吃药", "看病", "症状", "生理期",
            "姨妈期", "经期", "孕期", "哺乳期", "卡路里", "体脂", "步数",
            "身体", "买药", "压力", "焦虑", "抑郁",
        },
        ServiceDomain.SCHEDULE: {
            "日程", "安排", "会议", "提醒", "待办", "行程", "预约", "时间表",
        },
        ServiceDomain.HOME: {
            "空调", "窗帘", "热水器", "扫地", "开灯", "关灯", "灯",
        },
        ServiceDomain.SHOPPING: {
            "购物", "商品", "比价", "订单", "下单", "打折", "买", "推荐",
            "清单", "性价比", "哪个好", "买什么", "好物", "种草", "划算",
            "优惠券", "促销", "满减", "折扣", "评测", "测评", "排行",
            "榜单", "值得买",
        },
        ServiceDomain.SEARCH: {
            "天气", "预报", "气温", "下雨", "空气质量", "带伞", "穿衣",
            "穿什么", "雾霾", "台风", "紫外线", "湿度",
        },
        ServiceDomain.ROUTE: {
            "导航", "路线", "怎么走", "怎么去", "驾车", "公交", "地铁",
            "距离", "多远", "多少公里", "自驾", "徒步",
        },
    }

    DOMAIN_TIEBREAKER = {
        (ServiceDomain.HEALTH, ServiceDomain.SHOPPING): ServiceDomain.HEALTH,
        (ServiceDomain.HEALTH, ServiceDomain.SEARCH): ServiceDomain.HEALTH,
        (ServiceDomain.HEALTH, ServiceDomain.ROUTE): ServiceDomain.HEALTH,
        (ServiceDomain.SCHEDULE, ServiceDomain.HEALTH): ServiceDomain.SCHEDULE,
        (ServiceDomain.HOME, ServiceDomain.HEALTH): ServiceDomain.HOME,
        (ServiceDomain.ROUTE, ServiceDomain.SEARCH): ServiceDomain.ROUTE,
    }

    CONFIDENCE_THRESHOLD = 0.5
    MULTI_INTENT_INDICATORS = ["并且", "同时", "然后", "并", "另外", "以及", "还有", "；", ";"]

    @classmethod
    def recognize(cls, text: str, context: SessionContext) -> IntentResult:
        scores = {}
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            strong = cls.STRONG_KEYWORDS.get(domain, set())
            score = sum(2 if kw in strong else 1 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = min(score / 4.0, 1.0)

        if not scores:
            if context.dialog_history:
                last_turn = context.dialog_history[-1]
                if last_turn.intent:
                    for domain in ServiceDomain:
                        if domain.value == last_turn.intent:
                            return IntentResult(
                                intents=[Intent(domain, text, confidence=0.6)],
                                confidence=0.6,
                                needs_clarification=True,
                                clarification_question=f"您是想继续询问{domain.value}相关的内容吗？请具体描述您的需求。"
                            )
            return IntentResult(
                needs_clarification=True,
                clarification_question="未能理解您的请求，请补充说明或换种方式表达。"
            )

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_domains) >= 2 and sorted_domains[0][1] == sorted_domains[1][1]:
            d1, d2 = sorted_domains[0][0], sorted_domains[1][0]
            winner = cls.DOMAIN_TIEBREAKER.get((d1, d2))
            if winner is None:
                winner = cls.DOMAIN_TIEBREAKER.get((d2, d1))
            if winner is not None:
                sorted_domains = [(winner, sorted_domains[0][1])] + \
                    [(d, s) for d, s in sorted_domains if d != winner]

        intents = []
        for domain, score in sorted_domains:
            if score >= 0.25:
                slots = cls._extract_slots(text, domain)
                intents.append(Intent(domain, text, slots=slots, confidence=score))

        if len(intents) >= 2:
            intents = [intents[0]]

        top_confidence = sorted_domains[0][1]
        if top_confidence < cls.CONFIDENCE_THRESHOLD and len(intents) == 1:
            return IntentResult(
                intents=intents,
                confidence=top_confidence,
                needs_clarification=True,
                clarification_question=f"您是想咨询{intents[0].domain.value}相关的问题吗？请更具体地描述您的需求。"
            )

        return IntentResult(intents=intents, confidence=top_confidence)

    @classmethod
    def split_multi_intent(cls, text: str) -> list:
        for indicator in cls.MULTI_INTENT_INDICATORS:
            if indicator in text:
                parts = re.split(rf'[{indicator}]|{indicator}', text)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    return parts
        return [text]

    @staticmethod
    def _extract_slots(text: str, domain: ServiceDomain) -> dict:
        slots = {}
        if domain == ServiceDomain.SCHEDULE:
            time_match = re.search(r'(\d+)[点时](\d*)分?', text)
            if time_match:
                slots["time"] = time_match.group(0)
            if "后天" in text:
                slots["date"] = "后天"
            elif "明天" in text:
                slots["date"] = "明天"
            elif "今天" in text:
                slots["date"] = "今天"
            else:
                m = re.search(r'(\d{1,2})月(\d{1,2})日', text)
                if m:
                    slots["date"] = f"{m.group(1)}月{m.group(2)}日"
            if any(k in text for k in ["安排", "记录", "添加", "创建"]):
                slots["action"] = "add"
            elif any(k in text for k in ["查询", "查看", "有什么", "哪些"]):
                slots["action"] = "query"
        elif domain == ServiceDomain.SEARCH:
            city = WeatherAgent._extract_city_name(text)
            if city != "北京":
                slots["city"] = city
            if any(k in text for k in ["明天", "后天", "未来", "几天", "预报"]):
                slots["weather_type"] = "forecast"
            elif "天气" in text or "气温" in text:
                slots["weather_type"] = "live"
        elif domain == ServiceDomain.HOME:
            devices = ["灯", "空调", "电视", "窗帘", "热水器"]
            for d in devices:
                if d in text:
                    slots["device"] = d
                    break
            if any(k in text for k in ["打开", "开"]):
                slots["action"] = "on"
            elif any(k in text for k in ["关闭", "关"]):
                slots["action"] = "off"
        elif domain == ServiceDomain.ROUTE:
            patterns = [
                r'从(.+?)(?:到|去|至)(.+?)(?:怎么|路线|多远|距离|需要|的|$)',
                r'(.+?)(?:到|去|至)(.+?)(?:怎么|路线|多远|距离|需要|的|$)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    slots["origin"] = match.group(1).strip()
                    dest = match.group(2).strip()
                    for stop_word in ["怎么走", "怎么去", "路线", "多远", "距离", "需要多久"]:
                        dest = dest.replace(stop_word, "").strip()
                    slots["destination"] = dest
                    break
            if any(k in text for k in ["步行", "走路", "徒步"]):
                slots["travel_mode"] = "walking"
            elif any(k in text for k in ["公交", "地铁", "乘车"]):
                slots["travel_mode"] = "transit"
            elif any(k in text for k in ["驾车", "开车", "自驾"]):
                slots["travel_mode"] = "driving"
        return slots


# ==========================================================================
# 八、编排引擎 (Orchestration Engine)
# ==========================================================================

class OrchestrationEngine:
    """多 Agent 编排与协同 — spec.md §5.2"""

    def __init__(self, registry: AgentRegistry, context_mgr: ContextManager,
                 tool_executor: ToolExecutor, audit: AuditLogger,
                 approval_mgr: ApprovalManager):
        self._registry = registry
        self._context_mgr = context_mgr
        self._tools = tool_executor
        self._audit = audit
        self._approval = approval_mgr
        self._executor = ThreadPoolExecutor(max_workers=5)

    def orchestrate(self, intents: list, session: Session,
                    user: User) -> OrchestrationTask:
        task = OrchestrationTask(
            task_id=str(uuid.uuid4())[:8],
            session_id=session.session_id,
            intents=intents
        )

        sub_tasks = []
        for i, intent in enumerate(intents):
            agents = self._registry.discover(intent.domain, user.permission_level)
            if not agents:
                self._audit.log(user.user_id, "Agent发现", intent.domain.value,
                                "失败-无可用Agent")
                continue
            agent = agents[0]
            depends = [sub_tasks[i-1].sub_task_id] if i > 0 and self._has_dependency(intents[i], intents[i-1]) else []
            sub_task = SubTask(
                sub_task_id=str(uuid.uuid4())[:8],
                target_agent_id=agent.agent_id,
                intent=intent,
                depends_on=depends
            )
            sub_tasks.append(sub_task)
            task.add_sub_task(sub_task)

        if not sub_tasks:
            task.status = TaskStatus.FAILED
            return task

        task.status = TaskStatus.RUNNING
        self._audit.log(user.user_id, "编排开始", f"{len(sub_tasks)}个子任务", "执行中",
                        {"task_id": task.task_id})

        completed_ids = set()
        failed_ids = set()

        while len(completed_ids) + len(failed_ids) < len(sub_tasks):
            ready = task.get_ready_tasks(completed_ids | failed_ids)
            if not ready:
                break

            parallel = [st for st in ready if not st.depends_on]
            serial = [st for st in ready if st.depends_on]

            if parallel:
                futures = {}
                for st in parallel:
                    st.status = TaskStatus.RUNNING
                    agent = self._registry.get(st.target_agent_id)
                    future = self._executor.submit(
                        agent.execute, st.intent, session.context,
                        task.shared_context, self._tools
                    )
                    futures[future] = st

                for future in as_completed(futures):
                    st = futures[future]
                    try:
                        st.result = future.result(timeout=30)
                        st.status = TaskStatus.COMPLETED if st.result.success else TaskStatus.FAILED
                        if st.result.success:
                            completed_ids.add(st.sub_task_id)
                            self._context_mgr.update_shared_context(task, st.sub_task_id, st.result.data)
                        else:
                            failed_ids.add(st.sub_task_id)
                            self._audit.log(st.target_agent_id, "Agent执行",
                                            st.intent.query, "失败", {"error": st.result.error})
                    except Exception as e:
                        st.status = TaskStatus.FAILED
                        st.result = SubTaskResult(st.sub_task_id, st.target_agent_id,
                                                  False, f"执行异常: {str(e)}", error=str(e))
                        failed_ids.add(st.sub_task_id)

            for st in serial:
                st.status = TaskStatus.RUNNING
                agent = self._registry.get(st.target_agent_id)
                for dep_id in st.depends_on:
                    dep_result = task.shared_context.get(dep_id)
                    if dep_result:
                        session.context.add_result(f"dep_{dep_id}", dep_result)
                try:
                    st.result = agent.execute(st.intent, session.context,
                                             task.shared_context, self._tools)
                    st.status = TaskStatus.COMPLETED if st.result.success else TaskStatus.FAILED
                    if st.result.success:
                        completed_ids.add(st.sub_task_id)
                        self._context_mgr.update_shared_context(task, st.sub_task_id, st.result.data)
                    else:
                        failed_ids.add(st.sub_task_id)
                except Exception as e:
                    st.status = TaskStatus.FAILED
                    st.result = SubTaskResult(st.sub_task_id, st.target_agent_id,
                                              False, f"执行异常: {str(e)}", error=str(e))
                    failed_ids.add(st.sub_task_id)

        if len(completed_ids) == len(sub_tasks):
            task.status = TaskStatus.COMPLETED
        elif len(completed_ids) > 0:
            task.status = TaskStatus.PARTIAL
        else:
            task.status = TaskStatus.FAILED

        self._audit.log(user.user_id, "编排完成", task.task_id, task.status.value,
                        {"completed": len(completed_ids), "failed": len(failed_ids)})
        return task

    @staticmethod
    def _has_dependency(intent_a: Intent, intent_b: Intent) -> bool:
        dep_pairs = {(ServiceDomain.SEARCH, ServiceDomain.SCHEDULE),
                     (ServiceDomain.SEARCH, ServiceDomain.HEALTH)}
        return (intent_a.domain, intent_b.domain) in dep_pairs

    @staticmethod
    def aggregate_results(task: OrchestrationTask) -> str:
        successful = [st for st in task.sub_tasks
                      if st.result and st.result.success]
        failed = [st for st in task.sub_tasks
                  if st.result and not st.result.success]

        if not successful and failed:
            return "❌ 所有服务执行失败，请稍后重试。"

        parts = []
        for st in successful:
            agent = st.target_agent_id
            parts.append(st.result.response)

        if failed:
            failed_names = [st.target_agent_id for st in failed]
            parts.append(f"\n⚠️ 部分服务暂时不可用（{', '.join(failed_names)}），已为您返回可用结果。")

        if len(successful) > 1:
            header = f"🔗 多Agent协同完成（{len(successful)}个Agent参与）：\n\n"
            return header + "\n\n".join(parts)
        return "\n\n".join(parts)


# ==========================================================================
# 九、系统初始化
# ==========================================================================


def init_system():
    audit = AuditLogger()
    approval_mgr = ApprovalManager(audit)
    context_mgr = ContextManager()

    tool_registry = ToolRegistry()
    tool_registry.register(Tool("weather_query", "天气查询", ToolType.QUERY))
    tool_registry.register(Tool("device_control", "设备控制", ToolType.CONTROL, is_sensitive=True))
    tool_registry.register(Tool("calendar_query", "日程查询", ToolType.QUERY))
    tool_registry.register(Tool("calendar_add", "日程添加", ToolType.CONTROL))
    tool_registry.register(Tool("product_search", "商品搜索", ToolType.QUERY))
    tool_registry.register(Tool("health_query", "健康查询", ToolType.QUERY))
    tool_registry.register(Tool("amap_geocode", "高德地理编码", ToolType.QUERY))
    tool_registry.register(Tool("amap_direction_driving", "高德驾车路线", ToolType.QUERY, timeout_seconds=15))
    tool_registry.register(Tool("amap_direction_walking", "高德步行路线", ToolType.QUERY, timeout_seconds=15))
    tool_registry.register(Tool("amap_direction_transit", "高德公交路线", ToolType.QUERY, timeout_seconds=15))
    tool_registry.register(Tool("amap_distance", "高德距离测量", ToolType.QUERY))
    tool_executor = ToolExecutor(tool_registry, audit)

    agent_registry = AgentRegistry(audit)
    agent_registry.register(ScheduleAgent())
    agent_registry.register(HealthAgent())
    agent_registry.register(HomeAgent())
    agent_registry.register(ShoppingAgent())
    agent_registry.register(WeatherAgent())
    agent_registry.register(RouteAgent())

    engine = OrchestrationEngine(agent_registry, context_mgr, tool_executor, audit, approval_mgr)

    audit.log("system", "系统启动", "多Agent协同服务", "成功")
    return {
        "audit": audit, "approval": approval_mgr, "context": context_mgr,
        "tools": tool_registry, "tool_executor": tool_executor,
        "agents": agent_registry, "engine": engine
    }


# ==========================================================================
# 十、Streamlit 界面
# ==========================================================================

def init_session_state(sys_components):
    if "user" not in st.session_state:
        st.session_state.user = User(user_id="user_001", name="张三",
                                     permission_level=PermissionLevel.HIGH)
    if "session" not in st.session_state:
        st.session_state.session = Session(
            session_id=str(uuid.uuid4())[:8],
            user_id=st.session_state.user.user_id
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "task_history" not in st.session_state:
        st.session_state.task_history = []
    if "sys" not in st.session_state:
        st.session_state.sys = sys_components


def render_sidebar(sys_components):
    st.sidebar.title("🎛️ 控制面板")
    agents = sys_components["agents"]
    audit = sys_components["audit"]

    with st.sidebar.expander("🔑 API 配置状态", expanded=True):
        llm_ok = bool(LLM_API_KEY)
        amap_ok = bool(AMAP_API_KEY)
        st.write(f"{'✅' if llm_ok else '❌'} DeepSeek LLM API")
        if llm_ok:
            st.caption(f"Key: {LLM_API_KEY[:12]}... | Model: {LLM_MODEL}")
        else:
            st.caption("未配置！请设置环境变量或 Streamlit Secrets")
        st.write(f"{'✅' if amap_ok else '❌'} 高德地图 API")
        if amap_ok:
            st.caption(f"Key: {AMAP_API_KEY[:12]}...")
        if not llm_ok:
            st.warning("⚠️ LLM 未配置，购物/家居/天气助手将无法工作！")
            with st.expander("📋 配置方法"):
                st.code(
                    "方法1 - Streamlit Secrets (推荐):\n"
                    "在 .streamlit/secrets.toml 中添加:\n"
                    'llm_api_key = "sk-你的key"\n'
                    'llm_base_url = "https://api.deepseek.com/v1"\n'
                    'llm_model = "deepseek-chat"\n\n'
                    "方法2 - 环境变量:\n"
                    "export DEEPSEEK_API_KEY=sk-你的key\n\n"
                    "方法3 - config.json:\n"
                    '{"llm_api_key": "sk-你的key",\n'
                    ' "llm_base_url": "https://api.deepseek.com/v1",\n'
                    ' "llm_model": "deepseek-chat"}',
                    language="bash"
                )

    with st.sidebar.expander("🤖 Agent 状态", expanded=True):
        all_agents = agents.list_all()
        for agent in all_agents:
            col1, col2 = st.columns([3, 1])
            with col1:
                status_icon = "✅" if agent.status.value == AgentStatus.ENABLED.value else "⛔"
                st.write(f"{status_icon} {agent.name}")
                st.caption(f"领域: {agent.domain.value} | 版本: {agent.version} | "
                           f"权限: {agent.permission_level.name}")
                stats = agent.stats
                st.caption(f"调用: {stats['calls']}次 | 成功率: {stats['rate']}")
            with col2:
                if agent.status.value == AgentStatus.ENABLED.value:
                    if st.button("禁用", key=f"dis_{agent.agent_id}"):
                        agents.disable(agent.agent_id)
                        st.rerun()
                else:
                    if st.button("启用", key=f"en_{agent.agent_id}"):
                        agents.enable(agent.agent_id)
                        st.rerun()

    with st.sidebar.expander("🔧 工具注册表"):
        for tool in sys_components["tools"].list_tools():
            sensitive = "🔒" if tool.is_sensitive else "🔓"
            st.write(f"{sensitive} {tool.name} ({tool.tool_id})")
            st.caption(f"类型: {tool.tool_type.value} | 超时: {tool.timeout_seconds}s | "
                       f"重试: {tool.max_retries}次")

    with st.sidebar.expander("📝 审批队列"):
        pending = sys_components["approval"].get_pending()
        if not pending:
            st.info("暂无待审批操作")
        for record in pending:
            st.warning(f"ID: {record.approval_id}\n操作: {record.operation}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("通过", key=f"ap_{record.approval_id}"):
                    sys_components["approval"].process_approval(record.approval_id, True)
                    st.rerun()
            with col2:
                if st.button("拒绝", key=f"rj_{record.approval_id}"):
                    sys_components["approval"].process_approval(record.approval_id, False)
                    st.rerun()

    with st.sidebar.expander("👤 用户偏好"):
        prefs = sys_components["context"].get_all_preferences(st.session_state.user.user_id)
        if prefs:
            st.json(prefs)
        else:
            st.info("暂无已保存偏好")

    with st.sidebar.expander("📊 系统指标"):
        logs = audit.get_logs(limit=10000)
        total = len(logs)
        success = sum(1 for l in logs if "成功" in l.result or "完成" in l.result)
        rate = success / total * 100 if total > 0 else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("总操作", total)
        col2.metric("成功", success)
        col3.metric("成功率", f"{rate:.1f}%")

        agent_calls = sum(a._call_count for a in agents.list_all())
        st.metric("Agent总调用", agent_calls)
        st.metric("会话消息数", len(st.session_state.messages))


def render_chat_interface(sys_components):
    st.title("🏠 多 Agent 协同的智慧生活服务助手")
    st.caption("基于规格驱动开发 | Python + Streamlit | 多Agent编排引擎")

    agents = sys_components["agents"]
    engine = sys_components["engine"]
    audit = sys_components["audit"]
    context_mgr = sys_components["context"]

    active_agents = [a for a in agents.list_all() if a.status.value == AgentStatus.ENABLED.value]
    agent_cols = st.columns(max(1, len(active_agents)))
    for col, agent in zip(agent_cols, active_agents):
        col.metric(agent.name, agent.stats["calls"], delta=agent.stats["rate"])

    with st.expander("🏠 虚拟设备状态面板", expanded=True):
        device_state = VirtualDeviceState.get_state()
        dev_list = list(device_state.items())
        for i in range(0, len(dev_list), 4):
            row = dev_list[i:i + 4]
            cols = st.columns(len(row))
            for col, (device, info) in zip(cols, row):
                icon = info.get("icon", "🔹")
                power = info.get("power", False)
                with col:
                    if power:
                        st.metric(f"{icon} {device}", "开启")
                        if "temperature" in info:
                            st.caption(f"🌡️ {info['temperature']}°C {info.get('mode', '')}")
                        elif "brightness" in info:
                            st.caption(f"💡 亮度{info['brightness']}%")
                        elif "position" in info:
                            st.caption(f"📐 {info['position']}%")
                        elif "volume" in info:
                            st.caption(f"🔊 音量{info['volume']}")
                        elif "status" in info:
                            st.caption(f"⚙️ {info['status']}")
                    else:
                        st.metric(f"{icon} {device}", "关闭")

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "task" in message and message["task"]:
                with st.expander("🔍 编排详情", expanded=False):
                    task = message["task"]
                    st.write(f"任务ID: {task.task_id}")
                    st.write(f"状态: {task.status.value}")
                    st.write(f"子任务数: {len(task.sub_tasks)}")
                    for st_item in task.sub_tasks:
                        agent = agents.get(st_item.target_agent_id)
                        status_icon = {"已完成": "✅", "失败": "❌", "待执行": "⏳", "执行中": "🔄"}.get(st_item.status.value, "❓")
                        st.write(f"  {status_icon} {agent.name if agent else st_item.target_agent_id}: {st_item.status.value}")

    user_input = st.chat_input("请输入您的生活服务请求...")

    if user_input:
        session = st.session_state.session
        user = st.session_state.user

        is_safe, cleaned_input = InputSanitizer.sanitize(user_input)
        if not is_safe:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                st.error(f"🚫 {cleaned_input}")
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": f"🚫 {cleaned_input}"})
            return

        with st.chat_message("user"):
            st.markdown(cleaned_input)

        turn = DialogTurn(
            turn_id=str(uuid.uuid4())[:8],
            role="user",
            content=cleaned_input
        )
        session.context.add_turn(turn)
        session.update_activity()

        text_parts = IntentRecognizer.split_multi_intent(cleaned_input)
        all_intents = []
        for part in text_parts:
            result = IntentRecognizer.recognize(part, session.context)
            if result.needs_clarification and not result.intents:
                with st.chat_message("assistant"):
                    st.warning(f"🤔 {result.clarification_question}")
                st.session_state.messages.append({"role": "user", "content": cleaned_input})
                st.session_state.messages.append({"role": "assistant",
                                                    "content": f"🤔 {result.clarification_question}"})
                return
            all_intents.extend(result.intents)

        if not all_intents:
            with st.chat_message("assistant"):
                st.warning("🤔 未能理解您的请求，请补充说明或换种方式表达。")
            st.session_state.messages.append({"role": "user", "content": cleaned_input})
            st.session_state.messages.append({"role": "assistant",
                                                "content": "🤔 未能理解您的请求，请补充说明或换种方式表达。"})
            return

        with st.chat_message("assistant"):
            intent_names = [i.domain.value for i in all_intents]
            with st.spinner(f"正在调度 {', '.join(intent_names)} Agent..."):
                task = engine.orchestrate(all_intents, session, user)
                response = OrchestrationEngine.aggregate_results(task)

            st.markdown(response)

            with st.expander("🔍 编排详情", expanded=False):
                st.write(f"任务ID: {task.task_id}")
                st.write(f"状态: {task.status.value}")
                st.write(f"子任务数: {len(task.sub_tasks)}")
                for st_item in task.sub_tasks:
                    agent = agents.get(st_item.target_agent_id)
                    status_icon = {"已完成": "✅", "失败": "❌", "待执行": "⏳", "执行中": "🔄"}.get(st_item.status.value, "❓")
                    st.write(f"  {status_icon} {agent.name if agent else st_item.target_agent_id}: {st_item.status.value}")
                if task.shared_context:
                    st.write("共享上下文:")
                    st.json(task.shared_context)

        turn_reply = DialogTurn(
            turn_id=str(uuid.uuid4())[:8],
            role="assistant",
            content=response,
            intent=",".join(intent_names),
            agent_used=",".join(st_item.target_agent_id for st_item in task.sub_tasks)
        )
        session.context.add_turn(turn_reply)

        st.session_state.messages.append({"role": "user", "content": cleaned_input})
        st.session_state.messages.append({
            "role": "assistant", "content": response, "task": task
        })
        st.session_state.task_history.append(task)

        for intent in all_intents:
            if intent.slots:
                existing = context_mgr.get_preference(user.user_id, intent.domain)
                existing.update(intent.slots)
                context_mgr.save_preference(user.user_id, intent.domain, existing)

        home_affected = any(
            st_item.target_agent_id == "agent_home" and st_item.status.value == "已完成"
            for st_item in task.sub_tasks
        )
        if home_affected:
            st.rerun()


def render_admin_panel(sys_components):
    st.header("⚙️ 管理配置面板")
    agents = sys_components["agents"]
    audit = sys_components["audit"]

    tab1, tab2, tab3 = st.tabs(["Agent管理", "审计日志", "编排历史"])

    with tab1:
        st.subheader("Agent 注册表")
        all_agents = agents.list_all()
        agent_data = []
        for a in all_agents:
            agent_data.append({
                "Agent ID": a.agent_id,
                "名称": a.name,
                "服务领域": a.domain.value,
                "版本": a.version,
                "状态": a.status.value,
                "权限等级": a.permission_level.name,
                "可调用工具": ", ".join(a.allowed_tools),
                "能力声明": ", ".join(a.capabilities),
                "调用次数": a.stats["calls"],
                "成功率": a.stats["rate"]
            })
        st.dataframe(agent_data, use_container_width=True)

        st.subheader("Agent 启用/禁用")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**启用的 Agent**")
            for a in all_agents:
                if a.status.value == AgentStatus.ENABLED.value:
                    if st.button(f"禁用 {a.name}", key=f"admin_dis_{a.agent_id}"):
                        agents.disable(a.agent_id)
                        st.rerun()
        with col2:
            st.write("**禁用的 Agent**")
            disabled = [a for a in all_agents if a.status.value == AgentStatus.DISABLED.value]
            if not disabled:
                st.info("无禁用的Agent")
            for a in disabled:
                if st.button(f"启用 {a.name}", key=f"admin_en_{a.agent_id}"):
                    agents.enable(a.agent_id)
                    st.rerun()

    with tab2:
        st.subheader("审计日志")
        col1, col2 = st.columns([1, 3])
        with col1:
            log_limit = st.selectbox("显示条数", [50, 100, 200, 500], index=1)
            log_types = ["全部", "Agent注册", "Agent执行", "工具调用", "编排开始",
                         "编排完成", "审批发起", "审批处理", "系统启动"]
            log_filter = st.selectbox("操作类型", log_types)
        with col2:
            filter_type = None if log_filter == "全部" else log_filter
            logs = audit.get_logs(limit=log_limit, filter_type=filter_type)
            log_data = []
            for l in reversed(logs):
                log_data.append({
                    "时间": l.timestamp.strftime("%H:%M:%S"),
                    "操作人": l.operator,
                    "操作类型": l.operation_type,
                    "目标": l.target,
                    "结果": l.result,
                    "日志ID": l.log_id
                })
            if log_data:
                st.dataframe(log_data, use_container_width=True, height=400)
            else:
                st.info("暂无审计日志")

    with tab3:
        st.subheader("编排任务历史")
        if st.session_state.task_history:
            for task in reversed(st.session_state.task_history[-20:]):
                with st.expander(f"任务 {task.task_id} | {task.status.value} | {task.created_at.strftime('%H:%M:%S')}"):
                    st.write(f"**意图**: {', '.join(i.domain.value for i in task.intents)}")
                    st.write(f"**子任务数**: {len(task.sub_tasks)}")
                    for st_item in task.sub_tasks:
                        agent = agents.get(st_item.target_agent_id)
                        name = agent.name if agent else st_item.target_agent_id
                        status_icon = {"已完成": "✅", "失败": "❌", "待执行": "⏳"}.get(st_item.status.value, "❓")
                        st.write(f"  {status_icon} {name}: {st_item.status.value}")
                        if st_item.result:
                            st.text(st_item.result.response[:200])
                    if task.shared_context:
                        st.write("**共享上下文**:")
                        st.json(task.shared_context)
        else:
            st.info("暂无编排任务历史")


def render_monitoring_panel(sys_components):
    st.header("📈 系统监控面板")
    agents = sys_components["agents"]
    audit = sys_components["audit"]

    logs = audit.get_logs(limit=10000)
    total = len(logs)
    success = sum(1 for l in logs if "成功" in l.result or "完成" in l.result)
    failed = total - success

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总操作数", total)
    col2.metric("成功操作", success)
    col3.metric("失败操作", failed)
    col4.metric("成功率", f"{success/total*100:.1f}%" if total > 0 else "N/A")

    st.subheader("Agent 调用统计")
    agent_stats = []
    for a in agents.list_all():
        agent_stats.append({
            "Agent": a.name,
            "领域": a.domain.value,
            "状态": a.status.value,
            "调用次数": a.stats["calls"],
            "成功次数": a.stats["success"],
            "成功率": a.stats["rate"]
        })
    st.dataframe(agent_stats, use_container_width=True)

    st.subheader("操作类型分布")
    type_counts = defaultdict(int)
    for l in logs:
        type_counts[l.operation_type] += 1
    if type_counts:
        st.bar_chart(dict(type_counts))

    st.subheader("最近活动时间线")
    recent = logs[-15:]
    for l in reversed(recent):
        time_str = l.timestamp.strftime("%H:%M:%S")
        icon = "✅" if "成功" in l.result or "完成" in l.result else "❌"
        st.write(f"{time_str} {icon} [{l.operation_type}] {l.operator} → {l.target} ({l.result})")


def main():
    st.set_page_config(
        page_title="多Agent协同智慧生活服务助手",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "sys_components" not in st.session_state:
        st.session_state.sys_components = init_system()
    sys_components = st.session_state.sys_components

    init_session_state(sys_components)

    render_sidebar(sys_components)

    page = st.sidebar.radio(
        "页面导航",
        ["💬 对话服务", "⚙️ 管理配置", "📈 系统监控"],
        index=0
    )

    if page == "💬 对话服务":
        render_chat_interface(sys_components)
    elif page == "⚙️ 管理配置":
        render_admin_panel(sys_components)
    elif page == "📈 系统监控":
        render_monitoring_panel(sys_components)


if __name__ == "__main__":
    main()
