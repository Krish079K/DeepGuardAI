// lib/services/api_service.dart
// ------------------------------
// HTTP client for communicating with the FastAPI backend.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../config/app_config.dart';
import '../models/analysis_result.dart';

class ApiException implements Exception {
  final String message;
  final int? statusCode;
  ApiException(this.message, {this.statusCode});

  @override
  String toString() => 'ApiException($statusCode): $message';
}

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  String get _base => AppConfig.apiBaseUrl;

  // ── Health check ──────────────────────────────────────────────────────
  Future<bool> healthCheck() async {
    try {
      final res = await http
          .get(Uri.parse('$_base/health'))
          .timeout(AppConfig.connectTimeout);
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Test specific URL & measure latency ──────────────────────────────
  Future<Map<String, dynamic>> testCustomUrl(String url) async {
    final sw = Stopwatch()..start();
    try {
      var u = url.trim();
      if (!u.startsWith('http://') && !u.startsWith('https://')) {
        u = 'http://$u';
      }
      u = u.replaceAll(RegExp(r'/+$'), '');
      final res = await http
          .get(Uri.parse('$u/health'))
          .timeout(const Duration(seconds: 7));
      sw.stop();
      if (res.statusCode == 200) {
        return {'ok': true, 'ms': sw.elapsedMilliseconds, 'url': u};
      }
      return {'ok': false, 'error': 'HTTP ${res.statusCode}', 'ms': sw.elapsedMilliseconds};
    } catch (e) {
      sw.stop();
      return {'ok': false, 'error': 'Connection failed', 'ms': sw.elapsedMilliseconds};
    }
  }


  // ── Analyze video ─────────────────────────────────────────────────────
  /// Upload [videoFile] to /analyze and return the result.
  /// Calls [onProgress] with bytes sent (0.0 → 1.0).
  Future<AnalysisResult> analyzeVideo(
    File videoFile, {
    void Function(double)? onProgress,
  }) async {
    final uri     = Uri.parse('$_base/analyze');
    final request = http.MultipartRequest('POST', uri);

    final stream = http.ByteStream(videoFile.openRead());
    final length = await videoFile.length();
    final multipart = http.MultipartFile(
      'file',
      stream,
      length,
      filename: videoFile.path.split(Platform.pathSeparator).last,
    );
    request.files.add(multipart);

    http.StreamedResponse streamed;
    try {
      streamed = await request.send().timeout(AppConfig.receiveTimeout);
    } on SocketException catch (e) {
      throw ApiException('Cannot connect to backend: ${e.message}');
    } on TimeoutException {
      throw ApiException('Request timed out — is the backend running?');
    }

    if (streamed.statusCode != 200) {
      final body = await streamed.stream.bytesToString();
      String msg;
      try {
        msg = (jsonDecode(body) as Map)['detail'] as String? ?? body;
      } catch (_) {
        msg = body;
      }
      throw ApiException(msg, statusCode: streamed.statusCode);
    }

    final body = await streamed.stream.bytesToString();
    final json = jsonDecode(body) as Map<String, dynamic>;
    return AnalysisResult.fromJson(json);
  }

  // ── History ───────────────────────────────────────────────────────────
  Future<List<AnalysisResult>> getHistory() async {
    final res = await http
        .get(Uri.parse('$_base/history'))
        .timeout(AppConfig.connectTimeout);
    _checkStatus(res);
    final list = jsonDecode(res.body) as List<dynamic>;
    return list
        .map((e) => AnalysisResult.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<AnalysisResult> getHistoryItem(int id) async {
    final res = await http
        .get(Uri.parse('$_base/history/$id'))
        .timeout(AppConfig.connectTimeout);
    _checkStatus(res);
    return AnalysisResult.fromJson(
        jsonDecode(res.body) as Map<String, dynamic>);
  }

  Future<void> deleteHistoryItem(int id) async {
    final res = await http
        .delete(Uri.parse('$_base/history/$id'))
        .timeout(AppConfig.connectTimeout);
    _checkStatus(res);
  }

  Future<void> deleteAnalysis(int id) => deleteHistoryItem(id);

  // ── Helpers ───────────────────────────────────────────────────────────
  void _checkStatus(http.Response res) {
    if (res.statusCode >= 400) {
      String msg;
      try {
        msg = (jsonDecode(res.body) as Map)['detail'] as String? ?? res.body;
      } catch (_) {
        msg = res.body;
      }
      throw ApiException(msg, statusCode: res.statusCode);
    }
  }
}
