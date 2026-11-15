"""JWT 鉴权：Authorization: Bearer <token>，HS256 验签，AUTH_ENABLED 开关。

- AUTH_ENABLED=false：直接放行（兼容现有无鉴权部署，默认值）。
- AUTH_ENABLED=true：强制验签 + 校验 userId；缺失/格式错/签名错/过期/缺字段 -> 401。
- 与 Java 后端对齐：HS256 + JWT_SECRET（UTF-8 原始字节作 HMAC key），不强制 token 带 exp。
"""
import os

import jwt
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

# 启动时读取，修改后重启服务生效
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
JWT_SECRET = os.getenv("JWT_SECRET", "")

if AUTH_ENABLED and not JWT_SECRET:
    raise RuntimeError("AUTH_ENABLED=true 但未配置 JWT_SECRET，拒绝启动")


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
        payload = jwt.decode(parts[1], JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Token 无效（{e.__class__.__name__}）")

    user_id = payload.get("userId")
    if user_id is None or (isinstance(user_id, str) and not user_id.strip()):
        raise HTTPException(status_code=401, detail="Token 缺少有效的 userId")

    return payload
