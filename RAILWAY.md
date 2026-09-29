# Railway deployment — v0.5.2

This version uses a root `Dockerfile`. Railway should print:

    Using detected Dockerfile!

during the build.

## Why

YOLO-World `set_classes()` needs a CLIP text encoder. Ultralytics can try to
install its CLIP fork automatically at runtime, but Railway's normal runtime
image may not contain the `git` executable.

v0.5.2 fixes that by:

1. installing `git` in a Docker build stage
2. installing `git+https://github.com/ultralytics/CLIP.git` during build
3. keeping `ultralytics-opencv-headless` for server/headless OpenCV
4. preloading YOLO-World and the CLIP text encoder during the image build
5. copying the downloaded model/cache into the runtime image

The runtime container therefore does not need to AutoUpdate Python packages.

## Deploy

Replace the repository files with v0.5.2 and redeploy.

You do not need a new Railway variable for this fix.

Keep your API credentials in:

    Railway -> Service -> Variables

The real `.env` should remain outside GitHub.

## What to look for

Build logs should show Dockerfile detection and model preloading.

Runtime logs should NOT contain:

    ModuleNotFoundError: No module named 'clip'

or:

    Cannot find command 'git'
