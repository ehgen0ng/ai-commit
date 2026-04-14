import { GoogleGenAI } from '@google/genai';
import { ConfigKeys, ConfigurationManager } from './config';

/**
 * Creates and returns a Gemini API client instance.
 * Supports custom base URL for proxy/gateway usage.
 * @returns {GoogleGenAI} - The Gemini API client instance.
 * @throws {Error} - Throws an error if the API key is missing or empty.
 */
export function createGeminiAPIClient() {
  const configManager = ConfigurationManager.getInstance();
  const apiKey = configManager.getConfig<string>(ConfigKeys.GEMINI_API_KEY);
  const baseUrl = configManager.getConfig<string>(ConfigKeys.GEMINI_BASE_URL);

  if (!apiKey) {
    throw new Error('The GEMINI_API_KEY environment variable is missing or empty.');
  }

  return new GoogleGenAI({
    apiKey,
    ...(baseUrl ? { httpOptions: { baseUrl } } : {}),
  });
}

/**
 * Sends a generate content request to the Gemini API.
 * System messages are mapped to systemInstruction; user/assistant messages are mapped to contents.
 * @param {Array<{role: string, content: string}>} messages - The messages to send to the API.
 * @returns {Promise<string | undefined>} - A promise that resolves to the API response text.
 */
export async function GeminiAPI(messages: { role: string; content: string }[]) {
  try {
    const ai = createGeminiAPIClient();
    const configManager = ConfigurationManager.getInstance();
    const modelName = configManager.getConfig<string>(ConfigKeys.GEMINI_MODEL);
    const temperature = configManager.getConfig<number>(ConfigKeys.GEMINI_TEMPERATURE, 0.7);

    const systemInstruction = messages
      .filter(m => m.role === 'system')
      .map(m => m.content)
      .join('\n\n');

    const contents = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));

    const response = await ai.models.generateContent({
      model: modelName,
      contents,
      config: {
        temperature,
        ...(systemInstruction ? { systemInstruction } : {}),
      },
    });

    return response.text;
  } catch (error) {
    console.error('Gemini API call failed:', error);
    throw error;
  }
}
