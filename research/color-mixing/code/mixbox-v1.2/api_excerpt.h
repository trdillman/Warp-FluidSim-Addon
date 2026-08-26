// Source: https://github.com/scrtwpns/mixbox/tree/v1.2
// Immutable tag: v1.2, commit ace2baebae7750643e931f170fc953a75a5a6ed3
// Interface-only evidence excerpt.

void mixbox_srgb8_to_latent(unsigned char r, unsigned char g, unsigned char b,
                            float out_latent[MIXBOX_NUMLATENTS]);
void mixbox_latent_to_srgb8(const float latent[MIXBOX_NUMLATENTS],
                            unsigned char *r, unsigned char *g, unsigned char *b);

// Multiple colors can be mixed by concentration-weighting each latent vector,
// summing the latents, then converting the result back to RGB.
