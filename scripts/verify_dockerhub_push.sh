#!/bin/bash
# Docker Hub 镜像推送验证脚本
# 用法: ./scripts/verify_dockerhub_push.sh username imagename [tag]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

function print_info() {
    echo -e "${GREEN}[✓]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

function print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

function print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 参数检查
if [ -z "$1" ]; then
    echo "用法: $0 <docker_hub_username> [image_name] [tag]"
    echo "示例: $0 myuser cosyvoice-mms latest"
    exit 1
fi

USERNAME=$1
IMAGENAME=${2:-cosyvoice-mms}
TAG=${3:-latest}
FULL_IMAGE="${USERNAME}/${IMAGENAME}:${TAG}"

print_header "Docker Hub 镜像推送验证"

# 第一步：检查 Docker 是否安装
print_step "检查 Docker 安装..."
if command -v docker &> /dev/null; then
    print_info "Docker 已安装"
    docker --version
else
    print_error "Docker 未安装"
    exit 1
fi

# 第二步：检查 Docker 是否运行
print_step "检查 Docker 守护进程..."
if docker info &> /dev/null; then
    print_info "Docker 守护进程正在运行"
else
    print_error "Docker 守护进程未运行"
    exit 1
fi

# 第三步：检查本地镜像
print_step "检查本地镜像: $FULL_IMAGE"
if docker image inspect "$FULL_IMAGE" &> /dev/null; then
    print_info "本地镜像存在"
    
    # 获取镜像大小
    SIZE=$(docker images "$IMAGENAME" --format "{{.Size}}")
    print_info "镜像大小: $SIZE"
    
    # 获取镜像ID
    IMAGE_ID=$(docker images "$IMAGENAME" --format "{{.ID}}")
    print_info "镜像ID: $IMAGE_ID"
    
    # 获取创建时间
    CREATED=$(docker inspect "$FULL_IMAGE" | grep '"Created"' | head -1 | sed 's/.*"\([^"]*\)".*/\1/')
    print_info "创建时间: $CREATED"
else
    print_warn "本地镜像不存在: $FULL_IMAGE"
    print_warn "请先执行: docker build -t $FULL_IMAGE ."
fi

# 第四步：尝试拉取远程镜像
print_step "检查 Docker Hub 上的镜像: $FULL_IMAGE"
echo ""
echo "尝试从 Docker Hub 拉取镜像..."
echo "（这可能需要几分钟时间，取决于镜像大小）"
echo ""

if docker pull "$FULL_IMAGE" 2>&1 | tee /tmp/docker_pull.log; then
    print_info "✓ 镜像成功从 Docker Hub 拉取！"
    
    # 获取拉取后的镜像信息
    PULLED_SIZE=$(docker images "$IMAGENAME" --format "{{.Size}}")
    print_info "拉取的镜像大小: $PULLED_SIZE"
    
    # 第五步：运行镜像验证
    print_step "验证镜像功能..."
    
    # 创建测试容器
    TEST_CONTAINER=$(docker create "$FULL_IMAGE" echo "test" 2>/dev/null || echo "")
    
    if [ -n "$TEST_CONTAINER" ]; then
        print_info "容器创建成功"
        docker rm "$TEST_CONTAINER" &> /dev/null || true
    else
        print_warn "容器创建失败，但镜像已在 Docker Hub 上"
    fi
    
else
    print_error "✗ 镜像未能从 Docker Hub 拉取"
    print_error "可能原因："
    print_error "  1. 镜像未推送到 Docker Hub"
    print_error "  2. 镜像还未完成推送"
    print_error "  3. 用户名或标签不正确"
    exit 1
fi

# 第六步：检查 Docker Hub API
print_step "查询 Docker Hub API..."
echo ""

API_URL="https://hub.docker.com/v2/repositories/${USERNAME}/${IMAGENAME}/tags/${TAG}/"
print_info "API 端点: $API_URL"

RESPONSE=$(curl -s "$API_URL")

if echo "$RESPONSE" | grep -q '"name"'; then
    print_info "✓ Docker Hub API 查询成功"
    
    # 解析 JSON 响应
    echo "$RESPONSE" | jq '.' 2>/dev/null || true
else
    print_warn "无法从 Docker Hub API 查询镜像信息"
fi

# 第七步：生成报告
print_step "生成验证报告..."
echo ""

cat > /tmp/dockerhub_verification_report.txt << EOF
========================================
Docker Hub 镜像推送验证报告
========================================

镜像信息：
  完整名称: $FULL_IMAGE
  用户名: $USERNAME
  镜像名: $IMAGENAME
  标签: $TAG

验证结果：
  本地镜像: $(docker image inspect "$FULL_IMAGE" &> /dev/null && echo "✓ 存在" || echo "✗ 不存在")
  Docker Hub: ✓ 已推送
  拉取测试: ✓ 通过
  
Docker Hub 链接：
  https://hub.docker.com/r/${USERNAME}/${IMAGENAME}
  https://hub.docker.com/r/${USERNAME}/${IMAGENAME}/tags/${TAG}

验证时间: $(date -u +'%Y-%m-%d %H:%M:%S UTC')
========================================
EOF

cat /tmp/dockerhub_verification_report.txt

# 第八步：提供后续建议
print_step "后续步骤..."
echo ""
echo "如果验证通过，你可以：" 
echo ""
echo "1. 在生产环境中部署："
echo "   docker pull $FULL_IMAGE"
echo "   docker run --gpus all -p 8080:8080 $FULL_IMAGE"
echo ""
echo "2. 查看镜像详情："
echo "   docker inspect $FULL_IMAGE"
echo ""
echo "3. 查看镜像历史："
echo "   docker history $FULL_IMAGE"
echo ""
echo "4. 在 Docker Hub 中查看："
echo "   https://hub.docker.com/r/${USERNAME}/${IMAGENAME}"
echo ""

print_header "验证完成"
print_info "✓ 镜像 $FULL_IMAGE 已成功验证"
echo ""
echo "验证报告已保存到: /tmp/dockerhub_verification_report.txt"
echo ""

exit 0
