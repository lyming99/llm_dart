import 'package:dio/dio.dart';

import 'dart:convert';

import '../../core/capability.dart';
import '../../core/llm_error.dart';
import '../../models/chat_models.dart';
import '../../models/tool_models.dart';
import '../../utils/reasoning_utils.dart';
import 'client.dart';
import 'config.dart';

/// DeepSeek Chat capability implementation
///
/// This module handles all chat-related functionality for DeepSeek providers,
/// including streaming, tool calling, and reasoning model support.
class DeepSeekChat implements ChatCapability {
  final DeepSeekClient client;
  final DeepSeekConfig config;

  // State tracking for stream processing
  bool _hasReasoningContent = false;
  String _lastChunk = '';
  final StringBuffer _thinkingBuffer = StringBuffer();

  DeepSeekChat(this.client, this.config);

  String get chatEndpoint => 'chat/completions';

  @override
  Future<ChatResponse> chatWithTools(
    List<ChatMessage> messages,
    List<Tool>? tools, {
    CancelToken? cancelToken,
  }) async {
    final requestBody = _buildRequestBody(messages, tools, false);
    final responseData = await client.postJson(
      chatEndpoint,
      requestBody,
      cancelToken: cancelToken,
    );
    return _parseResponse(responseData);
  }

  @override
  Stream<ChatStreamEvent> chatStream(
    List<ChatMessage> messages, {
    List<Tool>? tools,
    CancelToken? cancelToken,
  }) async* {
    final effectiveTools = tools ?? config.tools;
    final requestBody = _buildRequestBody(messages, effectiveTools, true);

    // Reset stream state
    _resetStreamState();

    // Create SSE stream
    final stream = client.postStreamRaw(
      chatEndpoint,
      requestBody,
      cancelToken: cancelToken,
    );

    await for (final chunk in stream) {
      final events = _parseStreamEvents(chunk);
      for (final event in events) {
        yield event;
      }
    }
  }

  @override
  Future<ChatResponse> chat(
    List<ChatMessage> messages, {
    CancelToken? cancelToken,
  }) async {
    return chatWithTools(messages, null, cancelToken: cancelToken);
  }

  @override
  Future<List<ChatMessage>?> memoryContents() async => null;

  @override
  Future<String> summarizeHistory(List<ChatMessage> messages) async {
    final prompt =
        'Summarize in 2-3 sentences:\n${messages.map((m) => '${m.role.name}: ${m.content}').join('\n')}';
    final request = [ChatMessage.user(prompt)];
    final response = await chat(request);
    final text = response.text;
    if (text == null) {
      throw const GenericError('no text in summary response');
    }
    return text;
  }

  /// Reset stream state (call this when starting a new stream)
  void _resetStreamState() {
    _hasReasoningContent = false;
    _lastChunk = '';
    _thinkingBuffer.clear();
  }

  /// Parse response from DeepSeek API
  DeepSeekChatResponse _parseResponse(Map<String, dynamic> responseData) {
    // Extract reasoning content from non-streaming response
    // Reference: https://api-docs.deepseek.com/guides/reasoning_model
    String? thinkingContent;

    final choices = responseData['choices'] as List?;
    if (choices != null && choices.isNotEmpty) {
      final message = choices.first['message'] as Map<String, dynamic>?;
      if (message != null) {
        // Extract reasoning_content field from the message
        thinkingContent = message['reasoning_content'] as String?;
      }
    }

    return DeepSeekChatResponse(responseData, thinkingContent);
  }

  /// Parse stream events from SSE chunks with reasoning support
  List<ChatStreamEvent> _parseStreamEvents(String chunk) {
    final events = <ChatStreamEvent>[];
    final lines = chunk.split('\n');

    for (final line in lines) {
      if (line.startsWith('data: ')) {
        final data = line.substring(6).trim();
        if (data == '[DONE]') {
          break;
        }

        try {
          final json = jsonDecode(data) as Map<String, dynamic>;
          final streamEvents = _parseStreamEventWithReasoning(
            json,
            _hasReasoningContent,
            _lastChunk,
            _thinkingBuffer,
          );

          // Update tracking variables using reasoning utils
          final delta = _getDelta(json);
          if (delta != null) {
            final reasoningResult = ReasoningUtils.checkReasoningStatus(
              delta: delta,
              hasReasoningContent: _hasReasoningContent,
              lastChunk: _lastChunk,
            );
            _hasReasoningContent = reasoningResult.hasReasoningContent;
            _lastChunk = reasoningResult.updatedLastChunk;
          }

          events.addAll(streamEvents);
        } catch (e) {
          // Skip malformed JSON chunks
          client.logger
              .warning('Failed to parse stream JSON: $data, error: $e');
          continue;
        }
      }
    }

    return events;
  }

  /// Parse individual stream event with reasoning support
  List<ChatStreamEvent> _parseStreamEventWithReasoning(
    Map<String, dynamic> json,
    bool hasReasoningContent,
    String lastChunk,
    StringBuffer thinkingBuffer,
  ) {
    final events = <ChatStreamEvent>[];
    final choices = json['choices'] as List?;
    if (choices == null || choices.isEmpty) return events;

    final choice = choices.first as Map<String, dynamic>;
    final delta = choice['delta'] as Map<String, dynamic>?;
    if (delta == null) return events;

    // Handle reasoning content only for reasoning-capable models
    if (config.supportsReasoning) {
      final reasoningContent = ReasoningUtils.extractReasoningContent(delta);

      if (reasoningContent != null && reasoningContent.isNotEmpty) {
        thinkingBuffer.write(reasoningContent);
        _hasReasoningContent = true; // Update state
        events.add(ThinkingDeltaEvent(reasoningContent));
        return events;
      }
    }

    // Handle regular content
    final content = delta['content'] as String?;
    if (content != null && content.isNotEmpty) {
      // Update last chunk for reasoning detection
      _lastChunk = content;

      // Check reasoning status only for reasoning-capable models
      if (config.supportsReasoning) {
        final reasoningResult = ReasoningUtils.checkReasoningStatus(
          delta: delta,
          hasReasoningContent: _hasReasoningContent,
          lastChunk: lastChunk,
        );

        // Update state based on reasoning detection
        _hasReasoningContent = reasoningResult.hasReasoningContent;

        if (reasoningResult.isReasoningJustDone) {
          client.logger
              .fine('Reasoning phase completed, starting response phase');
        }
      }

      // Filter out thinking tags for models that use <think> tags
      if (ReasoningUtils.containsThinkingTags(content)) {
        // Extract thinking content and add to buffer
        final thinkMatch = RegExp(
          r'<think>(.*?)</think>',
          dotAll: true,
        ).firstMatch(content);
        if (thinkMatch != null) {
          final thinkingText = thinkMatch.group(1)?.trim();
          if (thinkingText != null && thinkingText.isNotEmpty) {
            thinkingBuffer.write(thinkingText);
            events.add(ThinkingDeltaEvent(thinkingText));
          }
        }
        // Don't emit content that contains thinking tags
        return events;
      }

      events.add(TextDeltaEvent(content));
    }

    // Handle tool calls
    final toolCalls = delta['tool_calls'] as List?;
    if (toolCalls != null && toolCalls.isNotEmpty) {
      final toolCall = toolCalls.first as Map<String, dynamic>;
      if (toolCall.containsKey('id') && toolCall.containsKey('function')) {
        try {
          events.add(ToolCallDeltaEvent(ToolCall.fromJson(toolCall)));
        } catch (e) {
          // Skip malformed tool calls
          client.logger.warning('Failed to parse tool call: $e');
        }
      }
    }

    // Check for finish reason
    final finishReason = choice['finish_reason'] as String?;
    if (finishReason != null) {
      final rawUsage = json['usage'];
      // Safely convert Map<dynamic, dynamic> to Map<String, dynamic>
      final Map<String, dynamic>? usage;
      if (rawUsage == null) {
        usage = null;
      } else if (rawUsage is Map<String, dynamic>) {
        usage = rawUsage;
      } else if (rawUsage is Map) {
        usage = Map<String, dynamic>.from(rawUsage);
      } else {
        usage = null;
      }

      final thinkingContent =
          thinkingBuffer.isNotEmpty ? thinkingBuffer.toString() : null;

      final response = DeepSeekChatResponse({
        'choices': [
          {
            'message': {'content': '', 'role': 'assistant'},
          },
        ],
        if (usage != null) 'usage': usage,
      }, thinkingContent);

      events.add(CompletionEvent(response));

      // Reset state after completion
      _resetStreamState();
    }

    return events;
  }

  /// Get delta from JSON response
  Map<String, dynamic>? _getDelta(Map<String, dynamic> json) {
    final choices = json['choices'] as List?;
    if (choices == null || choices.isEmpty) return null;

    final choice = choices.first as Map<String, dynamic>;
    return choice['delta'] as Map<String, dynamic>?;
  }

  /// Build request body for DeepSeek API
  Map<String, dynamic> _buildRequestBody(
    List<ChatMessage> messages,
    List<Tool>? tools,
    bool stream,
  ) {
    final apiMessages = <Map<String, dynamic>>[];

    // Add system message if configured
    if (config.systemPrompt != null) {
      apiMessages.add({'role': 'system', 'content': config.systemPrompt});
    }

    // Convert messages to DeepSeek format
    // ToolResultMessage must be expanded into individual 'tool' role messages
    // with tool_call_id, otherwise DeepSeek API returns:
    // "An assistant message with 'tool_calls' must be followed by tool messages
    // responding to each 'tool_call_id'"
    for (final message in messages) {
      if (message.messageType is ToolResultMessage) {
        final toolResults = (message.messageType as ToolResultMessage).results;
        for (final result in toolResults) {
          final resultContent = result.function.arguments.isNotEmpty
              ? result.function.arguments
              : (message.content.isNotEmpty ? message.content : 'Tool result');
          apiMessages.add({
            'role': 'tool',
            'tool_call_id': result.id,
            'content': resultContent,
          });
        }
      } else {
        apiMessages.add(_convertMessage(message));
      }
    }

    final body = <String, dynamic>{
      'model': config.model,
      'messages': apiMessages,
      'stream': stream,
    };

    // Check if using reasoning model
    final isReasonerModel = config.model == 'deepseek-reasoner';
    final isV4Thinking = config.isV4ThinkingModel;

    // Debug: log which messages have reasoning_content (V4 thinking mode)
    if (isV4Thinking) {
      for (var i = 0; i < apiMessages.length; i++) {
        final msg = apiMessages[i];
        if (msg.containsKey('reasoning_content')) {
          client.logger.fine(
            '[DeepSeek] Message[$i] role=${msg['role']} has reasoning_content '
            '(${(msg['reasoning_content'] as String?)?.length ?? 0} chars)',
          );
        }
      }

      // Debug: log assistant messages that lack reasoning_content (V4 requirement)
      var missingCount = 0;
      for (var i = 0; i < apiMessages.length; i++) {
        final msg = apiMessages[i];
        if (msg['role'] == 'assistant' &&
            !msg.containsKey('reasoning_content')) {
          missingCount++;
          if (missingCount <= 3) {
            client.logger.fine(
              '[DeepSeek] Message[$i] role=assistant MISSING reasoning_content '
              '(this will cause "reasoning_content must be passed back" error if '
              'this message was originally generated in thinking mode)',
            );
          }
        }
      }
      if (missingCount > 0) {
        client.logger.fine(
          '[DeepSeek] Total $missingCount assistant messages missing reasoning_content '
          '(thinking={"type": ${config.thinkingType ?? 'enabled'}}, reasoning_effort=${config.reasoningEffort ?? 'high'})',
        );
      }
    }

    // Add max_tokens (supported by all models)
    if (config.maxTokens != null) body['max_tokens'] = config.maxTokens;

    // DeepSeek-specific parameters
    // Reference: https://api-docs.deepseek.com/api/create-chat-completion

    if (isReasonerModel) {
      // deepseek-reasoner model restrictions
      // Reference: https://api-docs.deepseek.com/guides/reasoning_model
      // "Not Supported Parameters: temperature, top_p, presence_penalty, frequency_penalty, logprobs, top_logprobs"

      // logprobs and top_logprobs will trigger an error
      if (config.logprobs == true || config.topLogprobs != null) {
        client.logger.warning(
            'logprobs and top_logprobs are not supported by deepseek-reasoner model');
      }

      // temperature, top_p, presence_penalty, frequency_penalty have no effect but don't error
      if (config.temperature != null) {
        client.logger.info(
            'temperature parameter has no effect on deepseek-reasoner model');
      }
      if (config.topP != null) {
        client.logger
            .info('top_p parameter has no effect on deepseek-reasoner model');
      }
      if (config.frequencyPenalty != null) {
        client.logger.info(
            'frequency_penalty parameter has no effect on deepseek-reasoner model');
      }
      if (config.presencePenalty != null) {
        client.logger.info(
            'presence_penalty parameter has no effect on deepseek-reasoner model');
      }
    } else if (isV4Thinking) {
      // V4 models (deepseek-v4-flash, deepseek-v4-pro) thinking mode.
      // Reference: https://api-docs.deepseek.com/guides/thinking_mode
      //
      // Key rules:
      // - thinking mode is enabled by default; explicitly set it for clarity
      // - thinking mode does NOT support: temperature, top_p, presence_penalty, frequency_penalty
      //   (setting them won't error but they have no effect)
      // - reasoning_effort controls reasoning depth: 'high', 'medium', 'low'

      // Explicitly set thinking mode parameter
      final thinkingType = config.thinkingType ?? 'enabled';
      body['thinking'] = {'type': thinkingType};

      // Set reasoning effort (default: 'max' for best quality; 'high' for balanced)
      // Reference: https://api-docs.deepseek.com/guides/thinking_mode
      body['reasoning_effort'] = config.reasoningEffort ?? 'max';

      // Note: temperature, top_p, presence_penalty, frequency_penalty are NOT
      // supported in thinking mode. We deliberately skip them here (unlike
      // previous behavior which added them unconditionally at the top of
      // this method). Reference:
      // https://api-docs.deepseek.com/guides/thinking_mode#input-output-parameters
      if (config.topK != null) body['top_k'] = config.topK;
      if (config.logprobs != null) body['logprobs'] = config.logprobs;
      if (config.topLogprobs != null) body['top_logprobs'] = config.topLogprobs;
    } else {
      // For non-reasoner, non-V4 models (e.g. deepseek-chat), add all supported parameters
      if (config.temperature != null) body['temperature'] = config.temperature;
      if (config.topP != null) body['top_p'] = config.topP;
      if (config.topK != null) body['top_k'] = config.topK;
      if (config.logprobs != null) body['logprobs'] = config.logprobs;
      if (config.topLogprobs != null) body['top_logprobs'] = config.topLogprobs;
      if (config.frequencyPenalty != null) {
        body['frequency_penalty'] = config.frequencyPenalty;
      }
      if (config.presencePenalty != null) {
        body['presence_penalty'] = config.presencePenalty;
      }
    }

    // response_format is supported by both models
    if (config.responseFormat != null) {
      body['response_format'] = config.responseFormat;
    }

    // Add tools if provided
    final effectiveTools = tools ?? config.tools;
    if (effectiveTools != null && effectiveTools.isNotEmpty) {
      body['tools'] = effectiveTools.map((t) => t.toJson()).toList();

      final effectiveToolChoice = config.toolChoice;
      if (effectiveToolChoice != null) {
        body['tool_choice'] = effectiveToolChoice.toJson();
      }
    }

    return body;
  }

  /// Convert ChatMessage to DeepSeek format
  Map<String, dynamic> _convertMessage(ChatMessage message) {
    final result = <String, dynamic>{'role': message.role.name};

    // Add name field if present (DeepSeek is OpenAI-compatible)
    if (message.name != null) {
      result['name'] = message.name;
    }

    switch (message.messageType) {
      case TextMessage():
        result['content'] = message.content;
        break;
      case ToolUseMessage(toolCalls: final toolCalls):
        result['tool_calls'] = toolCalls.map((tc) => tc.toJson()).toList();
        // Ensure content is set for tool use messages (DeepSeek may require it)
        result['content'] = message.content;
        break;
      case ToolResultMessage():
        // Tool results are handled as separate messages in DeepSeek
        // This should be handled at a higher level
        result['content'] = message.content;
        break;
      default:
        result['content'] = message.content;
    }

    // Handle reasoning_content from extensions for thinking mode
    // IMPORTANT: The behavior differs between model generations:
    //
    // V4 models (deepseek-v4-flash, deepseek-v4-pro):
    //   - Thinking mode is enabled by default
    //   - reasoning_content MUST be passed back in multi-turn conversations
    //   - Missing reasoning_content causes: "The reasoning_content in the thinking mode must be passed back to the API"
    //   - Even when reasoning_content is empty, the field must be present for assistant messages
    //   - Reference: https://api-docs.deepseek.com/guides/thinking_mode
    //
    // Legacy deepseek-reasoner:
    //   - reasoning_content should NOT be included in input messages (causes 400 error)
    //   - Reference: https://api-docs.deepseek.com/guides/reasoning_model
    if (config.isV4ThinkingModel) {
      final deepseekData =
          message.getExtension<Map<String, dynamic>>('deepseek');
      if (deepseekData != null &&
          deepseekData.containsKey('reasoning_content')) {
        final reasoningContent = deepseekData['reasoning_content'] as String?;
        if (reasoningContent != null && reasoningContent.isNotEmpty) {
          result['reasoning_content'] = reasoningContent;
        }
      }

      // V4 thinking mode requires every replayed assistant message to include
      // the reasoning_content field, even when it is empty. Without this,
      // the API returns a 400 error in multi-turn conversations with tool calls.
      // Reference: https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
      if (message.role == ChatRole.assistant &&
          !result.containsKey('reasoning_content')) {
        result['reasoning_content'] = '';
        client.logger.fine(
          '[DeepSeek] _convertMessage: V4 assistant message missing reasoning_content, '
          'filled with empty string to prevent 400 error. '
          'hasExtension=${message.hasExtension('deepseek')}, '
          'extensions=${message.extensions.keys.toList()}',
        );
      }
    }

    return result;
  }
}

/// DeepSeek chat response implementation
class DeepSeekChatResponse implements ChatResponse {
  final Map<String, dynamic> _rawResponse;
  final String? _thinkingContent;

  DeepSeekChatResponse(this._rawResponse, [this._thinkingContent]);

  @override
  String? get text {
    final choices = _rawResponse['choices'] as List?;
    if (choices == null || choices.isEmpty) return null;

    final message = choices.first['message'] as Map<String, dynamic>?;
    return message?['content'] as String?;
  }

  @override
  List<ToolCall>? get toolCalls {
    final choices = _rawResponse['choices'] as List?;
    if (choices == null || choices.isEmpty) return null;

    final message = choices.first['message'] as Map<String, dynamic>?;
    final toolCalls = message?['tool_calls'] as List?;

    if (toolCalls == null) return null;

    return toolCalls
        .map((tc) => ToolCall.fromJson(tc as Map<String, dynamic>))
        .toList();
  }

  @override
  UsageInfo? get usage {
    final rawUsage = _rawResponse['usage'];
    if (rawUsage == null) return null;

    // Safely convert Map<dynamic, dynamic> to Map<String, dynamic>
    final Map<String, dynamic> usageData;
    if (rawUsage is Map<String, dynamic>) {
      usageData = rawUsage;
    } else if (rawUsage is Map) {
      usageData = Map<String, dynamic>.from(rawUsage);
    } else {
      return null;
    }

    return UsageInfo.fromJson(usageData);
  }

  @override
  String? get thinking => _thinkingContent;

  @override
  String toString() {
    final textContent = text;
    final calls = toolCalls;
    final thinkingContent = thinking;

    final parts = <String>[];

    if (thinkingContent != null) {
      parts.add('Thinking: $thinkingContent');
    }

    if (calls != null) {
      parts.add(calls.map((c) => c.toString()).join('\n'));
    }

    if (textContent != null) {
      parts.add(textContent);
    }

    return parts.join('\n');
  }
}
