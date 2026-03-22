from typing import List

from auth import auth
from auth.auth import Token, get_current_active_user, get_password_hash
from core import models
from core.database import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

_BCRYPT_MAX_PASSWORD_BYTES = 72


def _validate_password_length(password: str) -> None:
    """Validate password length for bcrypt.

    bcrypt only uses the first 72 bytes of the password; longer passwords can
    raise errors or be silently truncated depending on backend.
    """
    if len(password.encode("utf-8")) > _BCRYPT_MAX_PASSWORD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu quá dài (tối đa 72 bytes). Vui lòng đặt mật khẩu ngắn hơn.",
        )


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

    class Config:
        orm_mode = True


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    password: str


@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    _validate_password_length(user.password)

    # Check if username already exists in active users
    db_user = (
        db.query(models.User).filter(models.User.username == user.username).first()
    )
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Check if email already exists in active users
    db_email = db.query(models.User).filter(models.User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username, email=user.email, hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """Login and get access token."""
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user=Depends(get_current_active_user)):
    """Get current user information."""
    return current_user


@router.post("/change-password")
def change_password(
    req: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Đổi mật khẩu cho user hiện tại"""
    _validate_password_length(req.new_password)

    # Kiểm tra mật khẩu cũ
    if not auth.verify_password(req.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Mật khẩu cũ không đúng")
    # Cập nhật mật khẩu mới
    current_user.hashed_password = get_password_hash(req.new_password)
    db.commit()
    return {"msg": "Đổi mật khẩu thành công"}


@router.post("/delete-account")
def delete_account(
    req: DeleteAccountRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Xóa tài khoản của user hiện tại"""
    try:
        # Kiểm tra mật khẩu
        if not auth.verify_password(req.password, current_user.hashed_password):
            raise HTTPException(status_code=400, detail="Mật khẩu không đúng")

        # Xóa user
        db.delete(current_user)
        db.commit()

        return {"msg": "Tài khoản đã được xóa thành công"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Không thể xóa tài khoản: {str(e)}"
        )
