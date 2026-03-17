# Docker Deployment Plan for GLM-4.7-Flash

## Current State Analysis

### What We Have
- ✅ Working llama.cpp build with Blackwell CUDA support
- ✅ Functional server scripts (Q4 and Q8 versions)
- ✅ Model download automation
- ✅ Basic Dockerfile (needs updates)
- ✅ docker-compose.yml (needs updates)

### What Needs Work
- ⚠️ Dockerfile uses generic CUDA architectures (needs Blackwell fix)
- ⚠️ No multi-model support in Docker setup
- ⚠️ Network binding needs verification
- ⚠️ Model download strategy in container

## Docker Strategy Options

### Option 1: Pre-built Binary Image (Recommended)
**Approach**: Build llama.cpp in Dockerfile, download models separately

**Pros**:
- Fast container startup
- Models persist outside container
- Easy to switch between Q4/Q8
- Smaller image layers

**Cons**:
- Initial build takes ~5-10 minutes
- Need to manage models via volume mounts

**Image Size Estimate**:
- Base: nvidia/cuda:12.4.0-devel-ubuntu24.04 (~3GB)
- Build tools + dependencies: ~1GB
- llama.cpp compiled: ~700MB
- **Total Image**: ~4.7GB
- **Models (external volume)**: 18GB (Q4) or 35GB (Q8)

### Option 2: Model-Included Image
**Approach**: Bundle model inside Docker image

**Pros**:
- Single artifact deployment
- No external dependencies

**Cons**:
- Huge image size (22GB for Q4, 39GB for Q8)
- Can't easily switch models
- Slow push/pull times

**Image Size**:
- Q4 version: ~22GB
- Q8 version: ~39GB

## Recommended Approach: Option 1

### Build Time Estimate
1. **Image Build**: 5-10 minutes (one-time)
   - CUDA base image pull: 2-3 min
   - Install dependencies: 1-2 min
   - Compile llama.cpp: 3-5 min

2. **First Run Setup**: 10-30 minutes (one-time)
   - Model download Q4: 10-15 min
   - Model download Q8: 15-30 min

3. **Subsequent Starts**: <10 seconds

### Storage Requirements

**On Host**:
```
Docker Image:        ~4.7 GB
Models Directory:
  - Q4_K_XL:        ~18 GB
  - Q8_K_XL:        ~35 GB
  - Both:           ~53 GB
```

**Total**: ~58 GB for complete setup

### Resource Usage

**CPU**: Auto-detect (uses all available)
**GPU**: Single GPU required (97.8GB VRAM available)
**RAM**: ~2-4 GB (model in VRAM)
**VRAM**:
- Q4: ~20-25 GB (with 202K context)
- Q8: ~40-45 GB (with 202K context)

## Docker Compose Architecture

### Service Structure
```yaml
services:
  glm-q4:      # 4-bit quantization (fast)
  glm-q8:      # 8-bit quantization (quality)
```

### Port Mapping
- Q4 service: `11346:11346` (default)
- Q8 service: `11347:11346` (alternative port)

### Volume Strategy
```
./models:/app/models     # Shared model storage
```

## Implementation Checklist

### Phase 1: Dockerfile Updates
- [ ] Add Blackwell CUDA architecture fix
- [ ] Optimize layer caching
- [ ] Add healthcheck endpoint
- [ ] Multi-stage build for smaller image

### Phase 2: Docker Compose
- [ ] Separate Q4 and Q8 services
- [ ] Proper GPU allocation
- [ ] Volume mounts for models
- [ ] Network configuration
- [ ] Restart policies

### Phase 3: Helper Scripts
- [ ] `docker-build.sh` - Build image
- [ ] `docker-download-models.sh` - Download models via container
- [ ] `docker-start-q4.sh` - Start Q4 service
- [ ] `docker-start-q8.sh` - Start Q8 service

### Phase 4: Testing
- [ ] Build verification
- [ ] Model loading test
- [ ] Network accessibility test
- [ ] VRAM usage monitoring
- [ ] Multi-client load test

## Deployment Workflow

### Initial Setup (One-time)
```bash
# 1. Build Docker image (~5-10 min)
./docker-build.sh

# 2. Download models (~20-40 min for both)
./docker-download-models.sh

# 3. Start server (Q4 or Q8)
docker compose up glm-q4 -d
```

### Daily Operations
```bash
# Start
docker compose up glm-q4 -d

# Stop
docker compose down

# Switch to Q8
docker compose down
docker compose up glm-q8 -d

# Check logs
docker compose logs -f

# Check VRAM
docker exec glm-q4 nvidia-smi
```

## Network Configuration

### Internal Container
- Binds to: `0.0.0.0:11346`
- Accessible from: Any network interface

### Host Exposure
- Q4: `hordak:11346`
- Q8: `hordak:11347`

### OpenCode Config
No changes needed - same URL works for Docker or native:
```json
"baseURL": "http://hordak:11346/v1"
```

## Advantages of Docker Approach

1. **Isolation**: Clean environment, no system pollution
2. **Reproducibility**: Same setup anywhere
3. **Easy Updates**: Rebuild image for llama.cpp updates
4. **Multi-Instance**: Run Q4 and Q8 simultaneously on different ports
5. **Portability**: Move to different servers easily
6. **CI/CD Ready**: Can automate testing and deployment

## Disadvantages

1. **Build Time**: Initial setup takes longer
2. **Image Size**: ~5GB for the image
3. **Complexity**: Extra layer of abstraction
4. **Debugging**: Slightly harder than native

## Time Investment Estimate

**Initial Implementation**: 2-3 hours
- Dockerfile refinement: 1 hour
- Docker Compose setup: 30 min
- Helper scripts: 30 min
- Testing & debugging: 1 hour

**Documentation**: 30 min

**Total**: ~3-4 hours for complete Docker solution

## Risk Assessment

**Low Risk**:
- ✅ Base setup works (current Dockerfile exists)
- ✅ CUDA support verified in native build
- ✅ Network binding understood

**Medium Risk**:
- ⚠️ Blackwell architecture in Docker (needs testing)
- ⚠️ VRAM allocation with Docker GPU runtime

**Mitigation**:
- Test Docker GPU passthrough first
- Keep native installation as fallback
- Incremental migration (test Q4, then Q8)

## Recommendation

**Proceed with Docker?** ✅ Yes, but incrementally

**Approach**:
1. Start with updated Dockerfile for Q4 only
2. Test thoroughly with Q4 model
3. Extend to Q8 if Q4 works well
4. Run both native and Docker in parallel initially
5. Migrate fully once confidence is high

**Next Steps** (if approved):
1. Update Dockerfile with Blackwell fix
2. Create docker-compose with Q4 service
3. Test build and deployment
4. Measure VRAM usage
5. Add Q8 service
6. Document final workflow
