// lib/config/app_config.dart
// ---------------------------
// Dynamic configuration for backend server connection.
// Automatically persists user-specified URLs in SharedPreferences.

import 'package:shared_preferences/shared_preferences.dart';

class AppConfig {
  static const String defaultUrl = 'https://deepguardai-3yq7.onrender.com';
  static String _apiBaseUrl = defaultUrl;

  static String get apiBaseUrl => _apiBaseUrl;

  static const Duration connectTimeout = Duration(seconds: 15);
  static const Duration receiveTimeout = Duration(minutes: 5);
  static const int maxVideoSizeMb = 200;

  static const String _prefKey = 'deepguard_backend_url';

  /// Initialize config from SharedPreferences
  static Future<void> init() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getString(_prefKey);
      if (saved != null && saved.trim().isNotEmpty) {
        _apiBaseUrl = _cleanUrl(saved);
      }
    } catch (_) {}
  }

  /// Update and persist new backend URL
  static Future<void> setBaseUrl(String newUrl) async {
    final cleaned = _cleanUrl(newUrl);
    if (cleaned.isNotEmpty) {
      _apiBaseUrl = cleaned;
      try {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_prefKey, cleaned);
      } catch (_) {}
    }
  }

  /// Reset to default local IP
  static Future<void> resetToDefault() async {
    _apiBaseUrl = defaultUrl;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_prefKey);
    } catch (_) {}
  }

  static String _cleanUrl(String url) {
    var u = url.trim();
    if (!u.startsWith('http://') && !u.startsWith('https://')) {
      u = 'http://$u';
    }
    return u.replaceAll(RegExp(r'/+$'), '');
  }
}

