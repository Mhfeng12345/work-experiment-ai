"""JWT 鉴权：Authorization: Bearer <token>，HS256 验签，AUTH_ENABLED 开关。

- AUTH_ENABLED=false：直接放行（兼容现有无鉴权部署，默认值）。
- AUTH_ENABLED=true：强制验签 + 校验 userId；缺失/格式错/签名错/过期/缺字段 -> 401。
- 与 Java 后端对齐：HS256 + JWT_SECRET（UTF-8 原始字节作 HMAC key），不强制 token 带 exp。
- JWT_SECRET 支持多密钥：启动时自动判断——JSON 数组（["keyA","keyB"]）或多个密钥依次验签；单个字符串自动视为单元素数组（兼容模式）。
"""
import json
import os

import jwt
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

# 启动时读取，修改后重启服务生效
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
_RAW_JWT_SECRET = (os.getenv("JWT_SECRET", "") or "").strip()


def _parse_jwt_secrets(raw: str) -> list[str]:
    """启动时判断 JWT_SECRET 类型：以 [ 开头按 JSON 数组解析，其他非空值按单密钥兼容处理。"""
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JWT_SECRET JSON 数组格式错误：{e}") from e
        if not isinstance(parsed, list):
            raise RuntimeError('JWT_SECRET 需为 JSON 数组，如 ["keyA","keyB"]')
        return [str(s).strip() for s in parsed if str(s).strip()]
    return [raw]


JWT_SECRETS = _parse_jwt_secrets(_RAW_JWT_SECRET)

if AUTH_ENABLED and not JWT_SECRETS:
    raise RuntimeError("AUTH_ENABLED=true 但未配置 JWT_SECRET，拒绝启动")
if AUTH_ENABLED:
    mode = "JSON 数组" if _RAW_JWT_SECRET.startswith("[") else "单字符串兼容模式（建议改为数组）"
    print(f"[auth] AUTH_ENABLED=true，已加载 {len(JWT_SECRETS)} 个 JWT 密钥（{mode}）")


def _decode_with_any_secret(token: str) -> dict:
    """逐个密钥验签：任一通过即返回 payload；全部失败则抛出最后一个异常。"""
    expired = False
    for secret in JWT_SECRETS:
        try:
            return jwt.decode(token, secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            expired = True  # 签名正确但过期
        except jwt.InvalidTokenError:
            continue  # 换下一个密钥
    if expired:
        raise jwt.ExpiredSignatureError("Signature has expired")
    raise jwt.InvalidSignatureError("Signature verification failed")


def verify_token(authorization: str | None = Header(default=None)) -> dict:
    """FastAPI 依赖：鉴权通过返回 payload，失败抛 401；开关关闭时直接放行。"""
    if not AUTH_ENABLED:
        return {}

    if not authorization:
        raise HTTPException(
            status_code=401, detail="缺少 Authorization 头",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(
            status_code=401, detail="Authorization 格式错误，应为：Bearer <token>",
        )

    try:
        # HS256 强制验签（algorithms 白名单，杜绝 none 算法攻击）；
        # exp 存在才校验有效期（PyJWT 默认），兼容 Java 端不带 exp 的存量签发规则。
        payload = _decode_with_any_secret(parts[1])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Token 无效（{e.__class__.__name__}）")

    user_id = payload.get("userId")
    if user_id is None or (isinstance(user_id, str) and not user_id.strip()):
        raise HTTPException(status_code=401, detail="Token 缺少有效的 userId")

    return payload
