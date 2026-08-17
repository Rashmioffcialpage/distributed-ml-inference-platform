package ratelimit

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func okHandler(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }

func TestLimiter_AllowsWithinBurst(t *testing.T) {
	l := New(1, 5)
	h := l.Middleware(okHandler)

	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("X-API-Key", "client-a")
		rec := httptest.NewRecorder()
		h(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("request %d: want 200, got %d", i, rec.Code)
		}
	}
}

func TestLimiter_RejectsOverBurst(t *testing.T) {
	l := New(1, 3)
	h := l.Middleware(okHandler)

	var lastCode int
	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("X-API-Key", "client-b")
		rec := httptest.NewRecorder()
		h(rec, req)
		lastCode = rec.Code
	}
	if lastCode != http.StatusTooManyRequests {
		t.Fatalf("want 429 after exceeding burst, got %d", lastCode)
	}
}

func TestLimiter_TracksClientsSeparately(t *testing.T) {
	l := New(1, 1)
	h := l.Middleware(okHandler)

	reqA := httptest.NewRequest(http.MethodGet, "/", nil)
	reqA.Header.Set("X-API-Key", "client-a")
	recA := httptest.NewRecorder()
	h(recA, reqA)

	reqB := httptest.NewRequest(http.MethodGet, "/", nil)
	reqB.Header.Set("X-API-Key", "client-b")
	recB := httptest.NewRecorder()
	h(recB, reqB)

	if recA.Code != http.StatusOK || recB.Code != http.StatusOK {
		t.Fatalf("distinct clients should each get their own bucket: A=%d B=%d", recA.Code, recB.Code)
	}
}
