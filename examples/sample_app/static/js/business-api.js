/** Read browser-owned application state, tolerating stale local data. */
export function readApplicationState(key, fallback) {
    const value = localStorage.getItem(key);
    if (!value) return fallback;
    try { const parsed = JSON.parse(value); return parsed && typeof parsed === "object" ? parsed : fallback; } catch { return fallback; }
}

/** Application-owned login, conversation history and uploads. */
export function createBusinessAPI(baseURL = window.location.href) {
    const key = 'xtalk.sample.credentials';
    let credentials = readApplicationState(key, null);
    async function request(path, options = {}) {
        const response = await fetch(new URL(path, baseURL), {
            ...options,
            headers: { ...(credentials?.token ? {Authorization: `Bearer ${credentials.token}`} : {}), ...options.headers },
        });
        if (!response.ok) {
            if (response.status === 401) { credentials = null; localStorage.removeItem(key); }
            throw new Error(`Application request failed: ${response.status}`);
        }
        return response.json();
    }
    return {
        async login() {
            if (credentials?.expiresAt > Date.now() + 30000) return credentials;
            credentials = null;
            const result = await request('/api/auth/login', {method: 'POST'});
            const payload = result.access_token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
            const expiresAt = JSON.parse(atob(payload)).exp * 1000;
            credentials = {token: result.access_token, expiresAt, user: result.user};
            localStorage.setItem(key, JSON.stringify(credentials));
            return credentials;
        },
        create: () => request('/api/sessions', {method: 'POST'}),
        list: () => request('/api/sessions'),
        detail: id => request(`/api/sessions/${encodeURIComponent(id)}`),
        async upload(id, file) { const body = new FormData(); body.append('session_id', id); body.append('file', file); return request('/api/upload', {method: 'POST', body}); },
        authenticatedURL(url) { const result = new URL(url); result.searchParams.set('access_token', credentials.token); return result; },
    };
}
