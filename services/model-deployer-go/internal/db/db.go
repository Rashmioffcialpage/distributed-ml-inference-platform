// Package db wraps the Postgres connection and schema migration for
// model-deployer-go's model registry.
package db

import (
	"context"
	"embed"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schemaFS embed.FS

// Connect opens a pgx connection pool and applies the registry schema
// (idempotent CREATE TABLE IF NOT EXISTS — fine for a portfolio project;
// a production system would use a real migration tool like golang-migrate).
func Connect(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("connect postgres: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	schema, err := schemaFS.ReadFile("schema.sql")
	if err != nil {
		return nil, err
	}
	if _, err := pool.Exec(ctx, string(schema)); err != nil {
		return nil, fmt.Errorf("apply schema: %w", err)
	}
	return pool, nil
}
