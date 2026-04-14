import OpenAI from 'openai';
import { ConfigKeys, ConfigurationManager } from './config';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Creates and returns an OpenAI configuration object.
 * @returns {Object} - The OpenAI configuration object.
 * @throws {Error} - Throws an error if the API key is missing or empty.
 */
function getOpenAIConfig() {
  const configManager = ConfigurationManager.getInstance();
  const apiKey = configManager.getConfig<string>(ConfigKeys.OPENAI_API_KEY);
  const baseURL = configManager.getConfig<string>(ConfigKeys.OPENAI_BASE_URL);
  const apiVersion = configManager.getConfig<string>(ConfigKeys.AZURE_API_VERSION);

  if (!apiKey) {
    throw new Error('The OPENAI_API_KEY environment variable is missing or empty.');
  }

  const config: {
    apiKey: string;
    baseURL?: string;
    defaultQuery?: { 'api-version': string };
    defaultHeaders?: { 'api-key': string };
  } = {
    apiKey
  };

  if (baseURL) {
    config.baseURL = baseURL;
    if (apiVersion) {
      config.defaultQuery = { 'api-version': apiVersion };
      config.defaultHeaders = { 'api-key': apiKey };
    }
  }

  return config;
}

/**
 * Creates and returns an OpenAI API instance.
 * @returns {OpenAI} - The OpenAI API instance.
 */
export function createOpenAIApi() {
  const config = getOpenAIConfig();
  return new OpenAI(config);
}

/**
 * Sends a request to the OpenAI API using either the Chat Completions or the Responses endpoint,
 * depending on the OPENAI_API_MODE setting.
 * @param {ChatMessage[]} messages - The messages to send to the API.
 * @returns {Promise<string | undefined | null>} - A promise that resolves to the response text.
 */
export async function ChatGPTAPI(messages: ChatMessage[]) {
  const openai = createOpenAIApi();
  const configManager = ConfigurationManager.getInstance();
  const model = configManager.getConfig<string>(ConfigKeys.OPENAI_MODEL);
  const temperature = configManager.getConfig<number>(ConfigKeys.OPENAI_TEMPERATURE, 0.7);
  const apiMode = configManager.getConfig<string>(ConfigKeys.OPENAI_API_MODE, 'chat');
  const useStream = configManager.getConfig<boolean>(ConfigKeys.OPENAI_STREAM, false);

  if (apiMode === 'responses') {
    const systemParts = messages
      .filter(m => m.role === 'system')
      .map(m => m.content);
    const nonSystem = messages.filter(m => m.role !== 'system');

    const baseParams = {
      model,
      ...(systemParts.length ? { instructions: systemParts.join('\n\n') } : {}),
      input: nonSystem as any,
      temperature,
    };

    if (useStream) {
      const stream = await openai.responses.create({ ...baseParams, stream: true });
      let text = '';
      for await (const event of stream as any) {
        if (event?.type === 'response.output_text.delta' && typeof event.delta === 'string') {
          text += event.delta;
        }
      }
      return text;
    }

    const response = await openai.responses.create(baseParams);
    return response.output_text;
  }

  if (useStream) {
    const stream = await openai.chat.completions.create({
      model,
      messages: messages as any,
      temperature,
      stream: true,
    });
    let text = '';
    for await (const chunk of stream as any) {
      const delta = chunk?.choices?.[0]?.delta?.content;
      if (typeof delta === 'string') {
        text += delta;
      }
    }
    return text;
  }

  const completion = await openai.chat.completions.create({
    model,
    messages: messages as any,
    temperature,
  });

  return completion.choices[0]?.message?.content;
}
