"""REST API for the movie recommender.   Run from the repo root:  python api/app.py   (default http://127.0.0.1:5000)

Endpoints
  GET /api/health
  GET /api/recommend?user_id=<int>&k=<1-50, default 10>&method=<deployed|item_cf|user_cf|svd|svd_blend|popularity>&diversify=<none|safe|diverse>
  GET /api/similar?movie=<movie_id or title>&n=<1-20, default 5>
  GET /api/user/<id>/history?limit=<1-200, default 50>
  GET /api/explain?user_id=<int>&movie=<movie_id or title>
Errors are JSON: {"error": {"code": ..., "message": ..., "hint": ...}} with a suitable HTTP status."""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.serving import Service, METHODS, DIVERSIFY_MODES

MAX_K, MAX_N, MAX_LIMIT, MAX_ID = 50, 20, 200, 10 ** 9


class ApiError(Exception):
    def __init__(self, status, code, message, hint=None):
        super().__init__(message); self.status, self.code, self.message, self.hint = status, code, message, hint


def int_param(name, raw, lo, hi, default=None):
    if raw is None or raw == '':
        if default is None:
            raise ApiError(400, 'missing_parameter', f'Parameter "{name}" is required.', f'Example: ?{name}=1')
        return default
    if not re.fullmatch(r'-?\d+', raw.strip()):
        raise ApiError(400, 'invalid_parameter', f'Parameter "{name}" must be a whole number, got "{raw}".')
    v = int(raw)
    if not lo <= v <= hi:
        raise ApiError(400, 'out_of_range', f'Parameter "{name}" must be between {lo} and {hi}, got {v}.')
    return v


def choice_param(name, raw, allowed, default):
    if raw is None or raw == '':
        return default
    if raw not in allowed:
        raise ApiError(400, 'invalid_parameter', f'Parameter "{name}" must be one of {list(allowed)}, got "{raw}".')
    return raw


def create_app(service=None):
    app = Flask(__name__)
    app.json.sort_keys = False
    svc = service or Service.load()

    @app.get('/api/health')
    def health():
        return jsonify(svc.health())

    @app.get('/api/recommend')
    def recommend():
        a = request.args
        user_id = int_param('user_id', a.get('user_id'), 1, MAX_ID)
        k = int_param('k', a.get('k'), 1, MAX_K, default=10)
        method = choice_param('method', a.get('method'), METHODS, 'deployed')
        diversify = choice_param('diversify', a.get('diversify'), DIVERSIFY_MODES, None)
        return jsonify(svc.recommend(user_id, k, method, diversify))

    @app.get('/api/similar')
    def similar():
        movie = (request.args.get('movie') or '').strip()
        if not movie:
            raise ApiError(400, 'missing_parameter', 'Parameter "movie" is required (movie_id such as M001, or a title).', 'Example: ?movie=Inception')
        n = int_param('n', request.args.get('n'), 1, MAX_N, default=5)
        out = svc.similar(movie, n)
        if out is None:
            raise ApiError(404, 'movie_not_found', f'No movie matches "{movie}".', 'Use an exact title or a movie_id such as M001.')
        return jsonify(out)

    @app.get('/api/user/<user_id>/history')
    def history(user_id):
        uid = int_param('user_id', user_id, 1, MAX_ID)
        limit = int_param('limit', request.args.get('limit'), 1, MAX_LIMIT, default=50)
        out = svc.history(uid, limit)
        if out is None:
            raise ApiError(404, 'user_not_found', f'User {uid} has no rating history.', 'New users get popularity-based recommendations from /api/recommend.')
        return jsonify(out)

    @app.get('/api/explain')
    def explain():
        uid = int_param('user_id', request.args.get('user_id'), 1, MAX_ID)
        movie = (request.args.get('movie') or '').strip()
        if not movie:
            raise ApiError(400, 'missing_parameter', 'Parameter "movie" is required.')
        out = svc.explain(uid, movie)
        if out is None:
            raise ApiError(404, 'not_found', 'Unknown user or movie: no personalised explanation is available.')
        return jsonify(out)

    @app.errorhandler(ApiError)
    def api_error(e):
        body = {'error': {'code': e.code, 'message': e.message}}
        if e.hint:
            body['error']['hint'] = e.hint
        return jsonify(body), e.status

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({'error': {'code': 'not_found', 'message': 'Unknown route.', 'hint': 'See /api/health for the service status; endpoints are listed in docs/api.md.'}}), 404

    @app.errorhandler(405)
    def not_allowed(_):
        return jsonify({'error': {'code': 'method_not_allowed', 'message': 'This API is read-only: use GET.'}}), 405

    @app.errorhandler(Exception)
    def unexpected(e):
        if isinstance(e, HTTPException):                      # keep the real status (e.g. 400, 414) but answer in JSON
            return jsonify({'error': {'code': e.name.lower().replace(' ', '_'), 'message': e.description}}), e.code
        app.logger.exception(e)
        return jsonify({'error': {'code': 'internal_error', 'message': 'Something went wrong on the server.'}}), 500

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), debug=False)
