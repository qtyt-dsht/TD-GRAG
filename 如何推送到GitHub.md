# 如何推送 CulturLand-Check 到 GitHub

## 当前状态

✅ Git 仓库已初始化
✅ 所有文件已提交 (298 个文件, 387,470 行代码)
✅ 远程仓库已配置: https://github.com/qtyt-dsht/TD-GRAG.git
⚠️ 需要有效的 Personal Access Token 才能推送

## 问题原因

提供的 token 可能:
1. 权限不足 (没有勾选 `repo` 权限)
2. 已过期
3. 仓库 `qtyt-dsht/TD-GRAG` 的访问权限设置有问题

## 解决方案

### 方案 1: 重新创建 Personal Access Token (推荐)

#### 步骤 1: 创建新的 Token

1. 登录 GitHub 账号 `qtyt-dsht`
2. 访问: https://github.com/settings/tokens
3. 点击 **"Generate new token"** → **"Generate new token (classic)"**
4. 填写信息:
   - **Note**: `CulturLand-Check-Push` (token 名称)
   - **Expiration**: 选择有效期 (建议 90 days)
   - **Select scopes**: 
     - ✅ **repo** (勾选整个 repo 部分,包括所有子选项)
       - repo:status
       - repo_deployment
       - public_repo
       - repo:invite
       - security_events
5. 点击页面底部的 **"Generate token"**
6. **重要**: 立即复制生成的 token (格式: `ghp_xxxxxxxxxxxx`)

#### 步骤 2: 使用新 Token 推送

在命令行执行:

```bash
cd D:\个人论文\Christy\github_open_source\CulturLand-Check

# 设置远程 URL (将 YOUR_NEW_TOKEN 替换为刚才复制的 token)
git remote set-url origin https://YOUR_NEW_TOKEN@github.com/qtyt-dsht/TD-GRAG.git

# 推送到 GitHub
git push -u origin master
```

### 方案 2: 使用 GitHub Desktop (最简单,无需 Token)

1. **下载并安装 GitHub Desktop**
   - 访问: https://desktop.github.com/
   - 下载并安装

2. **登录账号**
   - 打开 GitHub Desktop
   - 点击 `File` → `Options` → `Accounts`
   - 点击 `Sign in` 登录 `qtyt-dsht` 账号

3. **添加本地仓库**
   - 点击 `File` → `Add local repository`
   - 选择路径: `D:\个人论文\Christy\github_open_source\CulturLand-Check`
   - 点击 `Add repository`

4. **推送到 GitHub**
   - 在 GitHub Desktop 中,点击 `Publish repository` 或 `Push origin`
   - 完成!

### 方案 3: 检查仓库是否存在

如果仓库 `qtyt-dsht/TD-GRAG` 不存在,需要先创建:

1. 访问: https://github.com/new
2. 填写:
   - **Repository name**: `TD-GRAG`
   - **Description**: `CulturLand-Check: Urban cultural land diagnosis framework`
   - **Public** 或 **Private** (根据需要选择)
   - ❌ 不要勾选 "Initialize this repository with a README"
3. 点击 **"Create repository"**
4. 然后使用方案 1 或方案 2 推送

### 方案 4: 使用 SSH (更安全,一次配置永久使用)

#### 步骤 1: 生成 SSH 密钥

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "1143326364@qq.com"

# 按 Enter 使用默认路径
# 可以设置密码或直接按 Enter 跳过

# 查看公钥
type %USERPROFILE%\.ssh\id_ed25519.pub
```

#### 步骤 2: 添加 SSH 密钥到 GitHub

1. 复制上一步显示的公钥内容 (以 `ssh-ed25519` 开头)
2. 访问: https://github.com/settings/keys
3. 点击 **"New SSH key"**
4. 填写:
   - **Title**: `CulturLand-Check-PC`
   - **Key**: 粘贴公钥内容
5. 点击 **"Add SSH key"**

#### 步骤 3: 修改远程 URL 并推送

```bash
cd D:\个人论文\Christy\github_open_source\CulturLand-Check

# 修改为 SSH URL
git remote set-url origin git@github.com:qtyt-dsht/TD-GRAG.git

# 推送
git push -u origin master
```

## 推送成功后

访问以下地址查看你的开源项目:
https://github.com/qtyt-dsht/TD-GRAG

## 常见问题

### Q: Token 权限不足怎么办?
A: 重新创建 token 时,确保勾选了完整的 `repo` 权限。

### Q: 推送时提示 "repository not found"?
A: 检查仓库是否存在,或者账号是否有权限访问该仓库。

### Q: 推送速度很慢?
A: 项目包含大量文件 (298 个文件, 387,470 行),首次推送可能需要几分钟。

### Q: 如何验证推送成功?
A: 推送成功后,命令行会显示:
```
Enumerating objects: xxx, done.
Counting objects: 100% (xxx/xxx), done.
...
To https://github.com/qtyt-dsht/TD-GRAG.git
 * [new branch]      master -> master
Branch 'master' set up to track remote branch 'master' from 'origin'.
```

## 项目信息

- **项目名称**: CulturLand-Check
- **仓库地址**: https://github.com/qtyt-dsht/TD-GRAG
- **本地路径**: `D:\个人论文\Christy\github_open_source\CulturLand-Check`
- **文件数量**: 298 个文件
- **代码行数**: 387,470 行
- **初始提交**: "Initial commit: CulturLand-Check open source release"

## 需要帮助?

如果以上方法都不行,可以:
1. 检查 GitHub 账号是否正常
2. 确认仓库 `qtyt-dsht/TD-GRAG` 是否存在且有写入权限
3. 尝试使用 GitHub Desktop (最简单可靠)
