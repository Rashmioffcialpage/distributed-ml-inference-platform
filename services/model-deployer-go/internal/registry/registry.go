// Package registry implements TeslaEdge's model registry: version
// bookkeeping, stage transitions (staging -> production -> archived),
// rollback and traffic-split percentages, backed by Postgres.
package registry

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rashmioffcialpage/distributed-ml-inference-platform/shared/pkg/models"
)

// ErrNotFound is returned when a requested model version doesn't exist.
var ErrNotFound = errors.New("model version not found")

// Registry is a Postgres-backed model version store.
type Registry struct {
	pool *pgxpool.Pool
}

// New wraps an existing connection pool.
func New(pool *pgxpool.Pool) *Registry {
	return &Registry{pool: pool}
}

// Register inserts (or replaces) a model version, initially in "staging".
func (r *Registry) Register(ctx context.Context, mv models.ModelVersion) error {
	metricsJSON, err := json.Marshal(mv.Metrics)
	if err != nil {
		return err
	}
	if mv.Stage == "" {
		mv.Stage = models.StageStaging
	}
	_, err = r.pool.Exec(ctx, `
		INSERT INTO model_versions (model_name, version, stage, precision, artifact_path, metrics, traffic_pct)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (model_name, version) DO UPDATE
		SET stage = $3, precision = $4, artifact_path = $5, metrics = $6, traffic_pct = $7
	`, mv.ModelName, mv.Version, mv.Stage, mv.Precision, mv.ArtifactPath, metricsJSON, mv.TrafficPct)
	return err
}

// Deploy promotes a version to production, giving it 100% traffic and
// archiving whichever version previously held production, then records a
// deployment_events row for auditability/rollback.
func (r *Registry) Deploy(ctx context.Context, modelName, version, reason string) error {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `
		UPDATE model_versions SET stage = $1, traffic_pct = 0
		WHERE model_name = $2 AND stage = $3
	`, models.StageArchived, modelName, models.StageProduction); err != nil {
		return err
	}

	tag, err := tx.Exec(ctx, `
		UPDATE model_versions SET stage = $1, traffic_pct = 100
		WHERE model_name = $2 AND version = $3
	`, models.StageProduction, modelName, version)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}

	if _, err := tx.Exec(ctx, `
		INSERT INTO deployment_events (model_name, version, action, reason) VALUES ($1, $2, 'deploy', $3)
	`, modelName, version, reason); err != nil {
		return err
	}

	return tx.Commit(ctx)
}

// Rollback finds the most recently archived version and promotes it back to
// production, demoting whatever is currently in production.
func (r *Registry) Rollback(ctx context.Context, modelName, reason string) (string, error) {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)

	var prevVersion string
	err = tx.QueryRow(ctx, `
		SELECT version FROM model_versions
		WHERE model_name = $1 AND stage = $2
		ORDER BY created_at DESC LIMIT 1
	`, modelName, models.StageArchived).Scan(&prevVersion)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", ErrNotFound
	}
	if err != nil {
		return "", err
	}

	if _, err := tx.Exec(ctx, `
		UPDATE model_versions SET stage = $1, traffic_pct = 0 WHERE model_name = $2 AND stage = $3
	`, models.StageArchived, modelName, models.StageProduction); err != nil {
		return "", err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE model_versions SET stage = $1, traffic_pct = 100 WHERE model_name = $2 AND version = $3
	`, models.StageProduction, modelName, prevVersion); err != nil {
		return "", err
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO deployment_events (model_name, version, action, reason) VALUES ($1, $2, 'rollback', $3)
	`, modelName, prevVersion, reason); err != nil {
		return "", err
	}

	return prevVersion, tx.Commit(ctx)
}

// SetTraffic sets an A/B traffic-split percentage for a version (0-100)
// without changing its stage, so a canary can run alongside production.
func (r *Registry) SetTraffic(ctx context.Context, modelName, version string, pct int) error {
	tag, err := r.pool.Exec(ctx, `
		UPDATE model_versions SET traffic_pct = $1 WHERE model_name = $2 AND version = $3
	`, pct, modelName, version)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

// Production returns the current production version of a model.
func (r *Registry) Production(ctx context.Context, modelName string) (models.ModelVersion, error) {
	return r.queryOne(ctx, `
		SELECT model_name, version, stage, precision, artifact_path, metrics, traffic_pct, created_at
		FROM model_versions WHERE model_name = $1 AND stage = $2
	`, modelName, models.StageProduction)
}

// List returns all versions of a model, newest first.
func (r *Registry) List(ctx context.Context, modelName string) ([]models.ModelVersion, error) {
	rows, err := r.pool.Query(ctx, `
		SELECT model_name, version, stage, precision, artifact_path, metrics, traffic_pct, created_at
		FROM model_versions WHERE model_name = $1 ORDER BY created_at DESC
	`, modelName)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []models.ModelVersion
	for rows.Next() {
		mv, err := scanVersion(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, mv)
	}
	return out, rows.Err()
}

type scannable interface {
	Scan(dest ...any) error
}

func scanVersion(row scannable) (models.ModelVersion, error) {
	var mv models.ModelVersion
	var metricsJSON []byte
	var createdAt time.Time
	if err := row.Scan(&mv.ModelName, &mv.Version, &mv.Stage, &mv.Precision, &mv.ArtifactPath, &metricsJSON, &mv.TrafficPct, &createdAt); err != nil {
		return mv, err
	}
	mv.CreatedAt = createdAt
	_ = json.Unmarshal(metricsJSON, &mv.Metrics)
	return mv, nil
}

func (r *Registry) queryOne(ctx context.Context, query string, args ...any) (models.ModelVersion, error) {
	row := r.pool.QueryRow(ctx, query, args...)
	mv, err := scanVersion(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return mv, ErrNotFound
	}
	return mv, err
}
