#!/bin/bash
# Docker Hub 构建和推送脚本
# 用法: ./ci/docker-hub-build.sh
# 环境变量:
#   DOCKER_HUB_USERNAME - Docker Hub 用户名
#   DOCKER_HUB_TOKEN - Docker Hub 访问令牌
#   CI_COMMIT_SHORT_SHA - 提交哈希（用于标签）
#   CI_COMMIT_BRANCH - 分支名称

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function validate_env() {
    print_info "验证环境变量..."
    
    if [ -z "$DOCKER_HUB_USERNAME" ]; then
        print_error "缺少环境变量: DOCKER_HUB_USERNAME"
        exit 1
    fi
    
    if [ -z "$DOCKER_HUB_TOKEN" ]; then
        print_error "缺少环境变量: DOCKER_HUB_TOKEN"
        exit 1
    fi
    
    print_info "环境变量验证通过"
}

function docker_login() {
    print_info "登录 Docker Hub..."
    
    echo "$DOCKER_HUB_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
    
    if [ $? -eq 0 ]; then
        print_info "Docker Hub 登录成功"
    else
        print_error "Docker Hub 登录失败"
        exit 1
    fi
}

function build_image() {
    print_info "开始构建镜像..."
    
    # 生成镜像标签
    local commit_sha="${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short HEAD)}"
    local branch="${CI_COMMIT_BRANCH:-main}"
    local timestamp=$(date +%Y%m%d)
    
    # 基础镜像名称
    local base_image="${DOCKER_HUB_USERNAME}/cosyvoice-mms"
    
    # 构建标签列表
    local tags=()
    tags+=("-t" "${base_image}:${commit_sha}")  # 提交哈希标签
    tags+=("-t" "${base_image}:${branch}-${timestamp}")  # 分支日期标签
    
    # 如果是主分支，添加 latest 标签
    if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
        tags+=("-t" "${base_image}:latest")
        print_info "将添加 'latest' 标签（主分支）"
    fi
    
    # 如果有 tag，使用 tag 作为版本号
    if [ ! -z "$CI_BUILD_TAG" ]; then
        tags+=("-t" "${base_image}:${CI_BUILD_TAG}")
        print_info "将添加版本标签: ${CI_BUILD_TAG}"
    fi
    
    print_info "镜像标签: $(printf '%s\n' "${tags[@]}" | grep -oP '(?<=-t ).*' | tr '\n' ', ')"
    
    # 构建镜像
    print_info "执行 docker build..."
    if docker build "${tags[@]}" .; then
        print_info "镜像构建成功"
    else
        print_error "镜像构建失败"
        exit 1
    fi
}

function scan_image() {
    print_info "扫描镜像安全漏洞（如已安装 grype）..."
    
    if command -v grype &> /dev/null; then
        local commit_sha="${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short HEAD)}"
        local base_image="${DOCKER_HUB_USERNAME}/cosyvoice-mms"
        local image="${base_image}:${commit_sha}"
        
        print_info "运行 grype 扫描: $image"
        
        if grype "$image" --fail-on critical; then
            print_info "安全扫描通过"
        else
            print_warn "检测到关键漏洞，但继续推送"
        fi
    else
        print_warn "grype 未安装，跳过安全扫描"
    fi
}

function push_image() {
    print_info "推送镜像到 Docker Hub..."
    
    local commit_sha="${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short HEAD)}"
    local branch="${CI_COMMIT_BRANCH:-main}"
    local base_image="${DOCKER_HUB_USERNAME}/cosyvoice-mms"
    
    local images_to_push=()
    images_to_push+=("${base_image}:${commit_sha}")
    images_to_push+=("${base_image}:${branch}-$(date +%Y%m%d)")
    
    if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
        images_to_push+=("${base_image}:latest")
    fi
    
    if [ ! -z "$CI_BUILD_TAG" ]; then
        images_to_push+=("${base_image}:${CI_BUILD_TAG}")
    fi
    
    for image in "${images_to_push[@]}"; do
        print_info "推送: $image"
        if docker push "$image"; then
            print_info "✓ 推送成功: $image"
        else
            print_error "✗ 推送失败: $image"
            exit 1
        fi
    done
}

function cleanup() {
    print_info "清理本地镜像..."
    
    local commit_sha="${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short HEAD)}"
    local branch="${CI_COMMIT_BRANCH:-main}"
    local base_image="${DOCKER_HUB_USERNAME}/cosyvoice-mms"
    
    # 删除本地镜像以节省磁盘空间
    docker rmi "${base_image}:${commit_sha}" || true
    docker rmi "${base_image}:${branch}-$(date +%Y%m%d)" || true
    docker rmi "${base_image}:latest" || true
    
    if [ ! -z "$CI_BUILD_TAG" ]; then
        docker rmi "${base_image}:${CI_BUILD_TAG}" || true
    fi
    
    print_info "清理完成"
}

function main() {
    print_info "=========================================="
    print_info "Docker Hub 镜像构建和推送"
    print_info "=========================================="
    
    validate_env
    docker_login
    build_image
    scan_image
    push_image
    cleanup
    
    print_info "=========================================="
    print_info "✓ 所有步骤完成成功"
    print_info "=========================================="
}

main
