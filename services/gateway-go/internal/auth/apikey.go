// Package auth implements TeslaEdge's gateway authentication: a static
// API-key allowlist checked against the X-API-Key header. This is
// intentionally simple (no OAuth/JWT infra) since the point of the gateway
// layer here is demonstrating auth + rate limiting exist at the edge, not
// building a full identity provider.
package auth

import "net/http"

// KeySet is a set of accepted API keys.
type KeySet map[string]bool

// NewKeySet builds a KeySet from a slice of keys.
func NewKeySet(keys []string) KeySet {
	s := make(KeySet, len(keys))
	for _, k := range keys {
		if k != "" {
			s[k] = true
		}
	}
	return s
}

// Middleware rejects requests whose X-API-Key header isn't in the KeySet. If
// the KeySet is empty (no keys configured), auth is a no-op — useful for
// local dev.
func (s KeySet) Middleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(s) == 0 {
			next(w, r)
			return
		}
		key := r.Header.Get("X-API-Key")
		if key == "" || !s[key] {
			http.Error(w, `{"error":"invalid or missing X-API-Key"}`, http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}
