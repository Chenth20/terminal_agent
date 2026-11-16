import time

import torch


PERIOD_SECONDS = 295
BURST_SECONDS = 50


def main():
    if not torch.cuda.is_available():
        raise SystemExit("当前环境未检测到 CUDA GPU")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    next_start = time.monotonic()

    with torch.inference_mode():
        while True:
            time.sleep(max(0, next_start - time.monotonic()))
            started = time.monotonic()

            a = torch.randn((8192, 8192), device=device, dtype=torch.bfloat16)
            b = torch.randn_like(a)
            out = torch.empty_like(a)
            torch.cuda.synchronize(device)

            deadline = time.monotonic() + BURST_SECONDS
            while time.monotonic() < deadline:
                torch.mm(a, b, out=out)
                torch.cuda.synchronize(device)

            del a, b, out
            torch.cuda.empty_cache()
            print(f"{time.strftime('%F %T')} 本轮完成", flush=True)
            next_start = started + PERIOD_SECONDS


if __name__ == "__main__":
    main()