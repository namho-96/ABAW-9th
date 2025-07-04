#!/usr/bin/env bash

# Multi-GPU 가속화 비디오 변환 스크립트 (NVENC)
set -e  # 오류 발생 시 즉시 중단

# ───────────────────────────────────────────────────────────────── #
# 색상 출력 함수
print_info()    { echo -e "\033[32m[INFO]\033[0m    $1"; }
print_warning() { echo -e "\033[33m[WARNING]\033[0m $1"; }
print_error()   { echo -e "\033[31m[ERROR]\033[0m   $1"; }
# ───────────────────────────────────────────────────────────────── #

# ───────────────────────────────────────────────────────────────── #
# NVIDIA GPU/NVENC 지원 확인
check_gpu_support() {
    if command -v nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_nvenc_support() {
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then
        return 0
    else
        return 1
    fi
}
# ───────────────────────────────────────────────────────────────── #

# ───────────────────────────────────────────────────────────────── #
# 시스템에 장착된 GPU 개수 반환 (nvidia-smi -L 줄 수)
detect_num_gpus() {
    local ngpu
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "$ngpu"
}
# ───────────────────────────────────────────────────────────────── #

# ───────────────────────────────────────────────────────────────── #
# 단일 파일 변환: GPU 모드면 지정된 GPU, 아니면 CPU x264
convert_single_video() {
    local input_file="$1"
    local output_file="$2"
    local gpu_id="$3"
    if [ "$GPU_MODE" = true ]; then
        print_info "GPU${gpu_id} 변환 중: $(basename "$input_file")"
        ffmpeg -y \
            -hwaccel cuda \
            -hwaccel_device "$gpu_id" \
            -hwaccel_output_format cuda \
            -i "$input_file" \
            -c:v h264_nvenc \
            -gpu "$gpu_id" \
            -preset fast \
            -b:v 5M -maxrate 8M -bufsize 10M \
            -c:a aac -b:a 128k \
            "$output_file" \
            -hide_banner -loglevel error
    else
        print_info "CPU 변환 중: $(basename "$input_file")"
        ffmpeg -y \
            -i "$input_file" \
            -c:v libx264 \
            -preset fast \
            -crf 23 \
            -c:a aac -b:a 128k \
            "$output_file" \
            -hide_banner -loglevel error
    fi

    if [ $? -eq 0 ]; then
        print_info "✅ 완료: $(basename "$output_file")"
    else
        print_error "❌ 실패: $(basename "$input_file")"
    fi
}
# ───────────────────────────────────────────────────────────────── #

# ───────────────────────────────────────────────────────────────── #
# 디렉토리 내 모든 비디오 병렬 처리
process_directory() {
    local video_dir="$1"
    [ ! -d "$video_dir" ] && { print_error "디렉토리 없음: $video_dir"; return; }

    # 출력 폴더
    local out_dir="$video_dir/converted_videos"
    mkdir -p "$out_dir"

    # 비디오 파일 목록
    mapfile -t video_files < <(find "$video_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.avi" -o -iname "*.mov" \))
    [ "${#video_files[@]}" -eq 0 ] && { print_warning "비디오 파일 없음: $video_dir"; return; }

    print_info "디렉토리: $video_dir — 총 ${#video_files[@]}개 파일 → $out_dir"
    local jobs=()

    for idx in "${!video_files[@]}"; do
        local file="${video_files[$idx]}"
        local name="$(basename "${file%.*}")"
        local out="$out_dir/${name}_converted.mp4"

        [ -f "$out" ] && { print_warning "이미 존재: $(basename "$out") — 스킵"; continue; }

        # 사용할 GPU 지정 (round-robin)
        local gpu_id=$(( idx % NGPU ))
        convert_single_video "$file" "$out" "$gpu_id" &
        jobs+=("$!")

        # 동시에 NGPU 개수만큼만 실행
        while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do
            wait -n
        done
    done

    # 남은 백그라운드 작업 대기
    wait
    print_info "=== 변환 완료: $video_dir ==="
}
# ───────────────────────────────────────────────────────────────── #

# ───────────────────────────────────────────────────────────────── #
# 스크립트 사용법
show_usage() {
    cat <<EOF
사용법: $0 [디렉토리1] [디렉토리2] ...

옵션:
  -h, --help    이 도움말 표시 후 종료

예시:
  $0                     # 기본 경로 두 개 처리
  $0 /videos/part1       # 단일 경로 지정
  $0 dir1 dir2 dir3      # 여러 경로 지정

기본 경로:
  ../../../dataset/dvd/DVD_Competition/Training/videos
  ../../../dataset/dvd/DVD_Competition/Validation/videos
EOF
}
# ───────────────────────────────────────────────────────────────── #

# ───────────────────────────────────────────────────────────────── #
# 메인
main() {
    print_info "=== Multi-GPU 비디오 변환 시작 ==="

    # 환경 설정: GPU/NVENC 지원 체크
    if check_gpu_support && check_nvenc_support; then
        GPU_MODE=true
        NGPU=$(detect_num_gpus)
        [ "$NGPU" -le 0 ] && NGPU=1
        print_info "GPU 모드 ON — NVIDIA GPU ${NGPU}개 사용"
    else
        GPU_MODE=false
        NGPU=1
        print_warning "GPU/NVENC 미지원 — CPU 모드"
    fi

    # 디렉토리 리스트 설정
    local base="../../../dataset/dvd/DVD_Competition"
    #dirs=( "$base/Training/videos" "$base/Validation/videos" )
    dirs=( "$base/Testing/videos")
    [ "$#" -gt 0 ] && dirs=("$@")

    # 각 디렉토리 처리
    for d in "${dirs[@]}"; do
        process_directory "$d"
    done

    print_info "=== 모든 작업 완료 ==="
}

# ───────────────────────────────────────────────────────────────── #
# 파라미터 검사
if [[ "$1" = "-h" || "$1" = "--help" ]]; then
    show_usage
    exit 0
fi

# 스크립트 실행
main "$@"
# ───────────────────────────────────────────────────────────────── #
