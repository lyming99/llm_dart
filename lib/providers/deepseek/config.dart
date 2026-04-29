import '../../models/tool_models.dart';
import '../../core/config.dart';
import '../../core/provider_defaults.dart';

/// DeepSeek provider configuration
///
/// This class contains all configuration options for the DeepSeek providers.
/// It's extracted from the main provider to improve modularity and reusability.
class DeepSeekConfig {
  final String apiKey;
  final String baseUrl;
  final String model;
  final int? maxTokens;
  final double? temperature;
  final String? systemPrompt;
  final Duration? timeout;

  final double? topP;
  final int? topK;
  final List<Tool>? tools;
  final ToolChoice? toolChoice;

  // DeepSeek-specific parameters
  final bool? logprobs;
  final int? topLogprobs;
  final double? frequencyPenalty;
  final double? presencePenalty;
  final Map<String, dynamic>? responseFormat;

  // V4 thinking mode parameters
  /// Controls thinking mode for V4 models ('enabled' or 'disabled').
  /// Only applicable to deepseek-v4-flash and deepseek-v4-pro.
  /// Reference: https://api-docs.deepseek.com/guides/thinking_mode
  final String? thinkingType;

  /// Reasoning effort level for V4 models ('high', 'medium', 'low').
  /// Only applicable to deepseek-v4-flash and deepseek-v4-pro.
  /// Reference: https://api-docs.deepseek.com/guides/thinking_mode
  final String? reasoningEffort;

  /// Reference to original LLMConfig for accessing extensions
  final LLMConfig? _originalConfig;

  const DeepSeekConfig({
    required this.apiKey,
    this.baseUrl = ProviderDefaults.deepseekBaseUrl,
    this.model = ProviderDefaults.deepseekDefaultModel,
    this.maxTokens,
    this.temperature,
    this.systemPrompt,
    this.timeout,
    this.topP,
    this.topK,
    this.tools,
    this.toolChoice,
    this.logprobs,
    this.topLogprobs,
    this.frequencyPenalty,
    this.presencePenalty,
    this.responseFormat,
    this.thinkingType,
    this.reasoningEffort,
    LLMConfig? originalConfig,
  }) : _originalConfig = originalConfig;

  /// Create DeepSeekConfig from unified LLMConfig
  factory DeepSeekConfig.fromLLMConfig(LLMConfig config) {
    return DeepSeekConfig(
      apiKey: config.apiKey!,
      baseUrl: config.baseUrl,
      model: config.model,
      maxTokens: config.maxTokens,
      temperature: config.temperature,
      systemPrompt: config.systemPrompt,
      timeout: config.timeout,
      topP: config.topP,
      topK: config.topK,
      tools: config.tools,
      toolChoice: config.toolChoice,
      // DeepSeek-specific parameters from extensions
      logprobs: config.getExtension<bool>('logprobs'),
      topLogprobs: config.getExtension<int>('top_logprobs'),
      frequencyPenalty: config.getExtension<double>('frequency_penalty'),
      presencePenalty: config.getExtension<double>('presence_penalty'),
      responseFormat:
          config.getExtension<Map<String, dynamic>>('response_format'),
      thinkingType: config.getExtension<String>('thinking_type'),
      reasoningEffort: config.getExtension<String>('reasoning_effort'),
      originalConfig: config,
    );
  }

  /// Get extension value from original config
  T? getExtension<T>(String key) => _originalConfig?.getExtension<T>(key);

  /// Get the original LLMConfig for HTTP configuration
  LLMConfig? get originalConfig => _originalConfig;

  /// Check if this model supports reasoning/thinking
  bool get supportsReasoning {
    // DeepSeek reasoner model and V4 models support reasoning
    // Reference: https://api-docs.deepseek.com/api/create-chat-completion
    return model == 'deepseek-reasoner' || isV4ThinkingModel;
  }

  /// Check if this is a V4 model with thinking mode
  /// V4 models (deepseek-v4-flash, deepseek-v4-pro) use thinking mode by default
  /// and require reasoning_content to be passed back in multi-turn conversations.
  /// Reference: https://api-docs.deepseek.com/api/create-chat-completion
  bool get isV4ThinkingModel {
    return model == 'deepseek-v4-flash' || model == 'deepseek-v4-pro';
  }

  /// Check if this model supports vision
  bool get supportsVision {
    // Currently no vision models available in DeepSeek API
    // Reference: https://api-docs.deepseek.com/api/list-models
    return false;
  }

  /// Check if this model supports tool calling
  bool get supportsToolCalling {
    // deepseek-chat, deepseek-reasoner, and V4 models support tool calling
    // Reference: https://api-docs.deepseek.com/guides/function_calling
    // Reference: https://api-docs.deepseek.com/guides/thinking_mode
    return model == 'deepseek-chat' || model == 'deepseek-reasoner' || isV4ThinkingModel;
  }

  /// Check if this model supports code generation
  bool get supportsCodeGeneration {
    // Both models can handle code generation tasks
    return model == 'deepseek-chat' || model == 'deepseek-reasoner';
  }

  DeepSeekConfig copyWith({
    String? apiKey,
    String? baseUrl,
    String? model,
    int? maxTokens,
    double? temperature,
    String? systemPrompt,
    Duration? timeout,
    double? topP,
    int? topK,
    List<Tool>? tools,
    ToolChoice? toolChoice,
    bool? logprobs,
    int? topLogprobs,
    double? frequencyPenalty,
    double? presencePenalty,
    Map<String, dynamic>? responseFormat,
    String? thinkingType,
    String? reasoningEffort,
  }) =>
      DeepSeekConfig(
        apiKey: apiKey ?? this.apiKey,
        baseUrl: baseUrl ?? this.baseUrl,
        model: model ?? this.model,
        maxTokens: maxTokens ?? this.maxTokens,
        temperature: temperature ?? this.temperature,
        systemPrompt: systemPrompt ?? this.systemPrompt,
        timeout: timeout ?? this.timeout,
        topP: topP ?? this.topP,
        topK: topK ?? this.topK,
        tools: tools ?? this.tools,
        toolChoice: toolChoice ?? this.toolChoice,
        logprobs: logprobs ?? this.logprobs,
        topLogprobs: topLogprobs ?? this.topLogprobs,
        frequencyPenalty: frequencyPenalty ?? this.frequencyPenalty,
        presencePenalty: presencePenalty ?? this.presencePenalty,
        responseFormat: responseFormat ?? this.responseFormat,
        thinkingType: thinkingType ?? this.thinkingType,
        reasoningEffort: reasoningEffort ?? this.reasoningEffort,
      );
}
