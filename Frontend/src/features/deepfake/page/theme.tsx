import { CONSOLE_PALETTE, type Palette } from "./palette";

/**
 * The page's colours, in one place.
 *
 * The deepfake page follows the rest of VoxLIT: one light console theme, no
 * theme switch. Components read their colours through this hook rather than
 * importing constants, so a future second theme only has to change here.
 */
export const usePalette = (): Palette => CONSOLE_PALETTE;
