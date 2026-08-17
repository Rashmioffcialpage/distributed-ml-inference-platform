package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func okHandler(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }

func TestKeySet_RejectsMissingKey(t *testing.T) {
	keys := NewKeySet([]string{"valid-key"})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()

	keys.Middleware(okHandler)(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", rec.Code)
	}
}

func TestKeySet_AcceptsValidKey(t *testing.T) {
	keys := NewKeySet([]string{"valid-key"})
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("X-API-Key", "valid-key")
	rec := httptest.NewRecorder()

	keys.Middleware(okHandler)(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200, got %d", rec.Code)
	}
}

func TestKeySet_EmptySetIsNoAuth(t *testing.T) {
	keys := NewKeySet(nil)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()

	keys.Middleware(okHandler)(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("want 200 (no keys configured = no-op auth), got %d", rec.Code)
	}
}
