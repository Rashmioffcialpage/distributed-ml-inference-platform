// Package handler implements model-deployer-go's REST API: register, list,
// deploy, rollback and traffic-split endpoints over the model registry.
package handler

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/rashmioffcialpage/distributed-ml-inference-platform/services/model-deployer-go/internal/registry"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
)

// Handler holds model-deployer-go's dependencies.
type Handler struct {
	Registry *registry.Registry
	Log      *slog.Logger
}

type registerRequest struct {
	ModelName    string             `json:"model_name"`
	Version      string             `json:"version"`
	Precision    string             `json:"precision"`
	ArtifactPath string             `json:"artifact_path"`
	Metrics      map[string]float64 `json:"metrics"`
}

// Register handles POST /v1/models/register.
func (h *Handler) Register(w http.ResponseWriter, r *http.Request) {
	var req registerRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.ModelName == "" || req.Version == "" || req.ArtifactPath == "" {
		writeError(w, http.StatusBadRequest, errors.New("model_name, version and artifact_path are required"))
		return
	}
	precision := models.Precision(req.Precision)
	if precision == "" {
		precision = models.PrecisionFP32
	}

	err := h.Registry.Register(r.Context(), models.ModelVersion{
		ModelName:    req.ModelName,
		Version:      req.Version,
		Precision:    precision,
		ArtifactPath: req.ArtifactPath,
		Metrics:      req.Metrics,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"status": "registered"})
}

// List handles GET /v1/models/{name}/versions.
func (h *Handler) List(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	versions, err := h.Registry.List(r.Context(), name)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, versions)
}

// Production handles GET /v1/models/{name}/production.
func (h *Handler) Production(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	mv, err := h.Registry.Production(r.Context(), name)
	if errors.Is(err, registry.ErrNotFound) {
		writeError(w, http.StatusNotFound, err)
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, mv)
}

type deployRequest struct {
	Version string `json:"version"`
	Reason  string `json:"reason"`
}

// Deploy handles POST /v1/models/{name}/deploy.
func (h *Handler) Deploy(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req deployRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Version == "" {
		writeError(w, http.StatusBadRequest, errors.New("version is required"))
		return
	}
	if err := h.Registry.Deploy(r.Context(), name, req.Version, req.Reason); err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, registry.ErrNotFound) {
			status = http.StatusNotFound
		}
		writeError(w, status, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deployed", "version": req.Version})
}

type rollbackRequest struct {
	Reason string `json:"reason"`
}

// Rollback handles POST /v1/models/{name}/rollback.
func (h *Handler) Rollback(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req rollbackRequest
	_ = json.NewDecoder(r.Body).Decode(&req)

	prev, err := h.Registry.Rollback(r.Context(), name, req.Reason)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, registry.ErrNotFound) {
			status = http.StatusNotFound
		}
		writeError(w, status, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "rolled_back", "version": prev})
}

type trafficRequest struct {
	Version    string `json:"version"`
	TrafficPct int    `json:"traffic_pct"`
}

// Traffic handles POST /v1/models/{name}/traffic for canary/A-B splits.
func (h *Handler) Traffic(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req trafficRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Version == "" {
		writeError(w, http.StatusBadRequest, errors.New("version is required"))
		return
	}
	if req.TrafficPct < 0 || req.TrafficPct > 100 {
		writeError(w, http.StatusBadRequest, errors.New("traffic_pct must be 0-100"))
		return
	}
	if err := h.Registry.SetTraffic(r.Context(), name, req.Version, req.TrafficPct); err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, registry.ErrNotFound) {
			status = http.StatusNotFound
		}
		writeError(w, status, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "updated", "version": req.Version, "traffic_pct": req.TrafficPct})
}

// Healthz handles GET /healthz.
func (h *Handler) Healthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}
