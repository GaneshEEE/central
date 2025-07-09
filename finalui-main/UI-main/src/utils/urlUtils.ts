export const getSpaceKeyFromURL = (): string | null => {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('space');
};

export const isSpaceConnected = (): boolean => {
  return getSpaceKeyFromURL() !== null;
};

export function getConfluenceSpaceAndPageFromUrl(): { space?: string; page?: string } {
  if (typeof window === 'undefined') return {};
  const params = new URLSearchParams(window.location.search);
  const space = params.get('space') || undefined;
  const page = params.get('page') || undefined;
  return { space, page };
} 