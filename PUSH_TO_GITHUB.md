# 如何推送到 GitHub

## 问题
GitHub 已经不再支持密码认证,需要使用 Personal Access Token (PAT)。

## 解决方案

### 方案 1: 使用 Personal Access Token (推荐)

1. **创建 Personal Access Token**
   - 访问: https://github.com/settings/tokens
   - 点击 "Generate new token" → "Generate new token (classic)"
   - 设置 Token 名称: `CulturLand-Check-Push`
   - 选择权限: 勾选 `repo` (完整仓库访问权限)
   - 点击 "Generate token"
   - **重要**: 复制生成的 token (只显示一次!)

2. **使用 Token 推送**
   ```bash
   cd github_open_source/CulturLand-Check
   
   # 设置远程 URL (将 YOUR_TOKEN 替换为你的 token)
   git remote set-url origin https://YOUR_TOKEN@github.com/qtyt-dsht/TD-GRAG.git
   
   # 推送
   git push -u origin master
   ```

### 方案 2: 使用 SSH (更安全)

1. **生成 SSH 密钥** (如果还没有)
   ```bash
   ssh-keygen -t ed25519 -C "qtyt.dsht@gmail.com"
   ```

2. **添加 SSH 密钥到 GitHub**
   - 复制公钥内容: `cat ~/.ssh/id_ed25519.pub`
   - 访问: https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴公钥内容并保存

3. **修改远程 URL 并推送**
   ```bash
   cd github_open_source/CulturLand-Check
   git remote set-url origin git@github.com:qtyt-dsht/TD-GRAG.git
   git push -u origin master
   ```

### 方案 3: 使用 GitHub Desktop (最简单)

1. 下载并安装 GitHub Desktop: https://desktop.github.com/
2. 登录 qtyt-dsht 账号
3. 添加本地仓库: `File` → `Add local repository` → 选择 `github_open_source/CulturLand-Check`
4. 点击 "Publish repository" 或 "Push origin"

### 方案 4: 使用 Git Credential Manager

Windows 系统可以使用 Git Credential Manager:
```bash
cd github_open_source/CulturLand-Check
git remote set-url origin https://github.com/qtyt-dsht/TD-GRAG.git
git push -u origin master
```
系统会弹出登录窗口,使用 qtyt-dsht 账号登录即可。

## 当前状态

✅ Git 仓库已初始化
✅ 所有文件已添加
✅ 初始提交已完成
✅ 远程仓库地址已配置
⏳ 等待推送 (需要认证)

## 推送后的验证

推送成功后,访问以下地址查看:
https://github.com/qtyt-dsht/TD-GRAG

## 注意事项

- 确保 GitHub 仓库 `qtyt-dsht/TD-GRAG` 已经创建
- 如果仓库不存在,需要先在 GitHub 上创建
- Token 和 SSH 密钥都要妥善保管
