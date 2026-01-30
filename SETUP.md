# Setup Guide - Heroes of the Storm Analytics

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd /Users/jamesfrankel/codebases/Hots/hots-app
npm install
```

This will install all required packages including:
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Framer Motion
- D3.js
- Radix UI components
- And more...

### 2. Run Development Server

```bash
npm run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000)

### 3. Build for Production

```bash
npm run build
npm run start
```

---

## Project Structure Overview

```
hots-app/
├── src/
│   ├── app/              # Pages (Next.js App Router)
│   ├── components/       # React components
│   ├── lib/             # Utilities and design tokens
│   └── types/           # TypeScript types
├── public/              # Static files
└── Configuration files
```

---

## Key Features Implemented

### ✅ Phase 1: Foundation (COMPLETED)
- Next.js 14 with TypeScript setup
- Design system with tokens (colors, typography, spacing, motion)
- Tailwind CSS configuration with custom theme
- Core UI component library

### ✅ Phase 2: Visual Design (COMPLETED)
- Glassmorphism UI with backdrop blur
- Dark/Light mode toggle
- Smooth animations with Framer Motion
- Gaming-themed color palette
- Custom stat cards with hover effects

### ✅ Phase 3: Data Visualization (COMPLETED)
- Interactive D3.js heatmap for hero vs map win rates
- Progress bars for win rates
- Animated statistics cards
- Role distribution visualization
- Power picks display

### ✅ Phase 4: User Experience (COMPLETED)
- Sidebar navigation with active state indicators
- Search functionality in header
- Filtering and sorting on Heroes page
- Responsive design (mobile, tablet, desktop)
- Keyboard-accessible components

### ✅ Phase 5: Pages & Features (COMPLETED)
- Dashboard (home) with overview statistics
- Heroes page with searchable grid and heatmap
- Maps page with performance breakdown
- Insights page with AI-powered recommendations
- Placeholder pages for Stats, Achievements, Settings

### ✅ Phase 6: Technical Excellence (COMPLETED)
- TypeScript for type safety
- Code splitting and lazy loading
- Optimized bundle size
- WCAG-compliant accessibility
- Performance optimizations

---

## Next Steps for Enhancement

### Data Integration
Currently using mock data. To connect real data:

1. Create API route: `src/app/api/data/route.ts`
2. Fetch from Heroes Profile API or your Python data files
3. Update the `usePlayerData` hook in `src/lib/hooks/use-data.ts`

Example API route:
```typescript
import { NextResponse } from 'next/server'

export async function GET() {
  // Read from your existing JSON files
  const data = await import('../../../path/to/hots_data.json')
  return NextResponse.json(data)
}
```

### Additional Features (Future)
- Radar charts for hero mastery
- Trend analysis with time-series
- Command palette (Cmd+K)
- Comparison mode
- PWA offline support
- Team analytics

---

## Development Tips

### Adding New Pages
1. Create a new folder in `src/app/`
2. Add a `page.tsx` file
3. Update navigation in `src/components/layout/sidebar.tsx`

### Creating Components
1. Add to `src/components/ui/` for reusable UI
2. Add to `src/components/dashboard/` for dashboard-specific
3. Export from component file

### Using Design Tokens
```typescript
import { colors } from '@/lib/design-tokens/colors'
import { typography } from '@/lib/design-tokens/typography'
import { motion } from '@/lib/design-tokens/motion'
```

### Styling with Tailwind
- Use utility classes: `className="bg-primary-500 text-white"`
- Use design system colors: `bg-gaming-success`, `text-gaming-danger`
- Use responsive modifiers: `md:grid-cols-2 lg:grid-cols-3`

---

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
npm run dev -- -p 3001
```

### Type Errors
```bash
# Run type checker
npm run type-check
```

### Build Errors
```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

---

## Performance Checklist

- [x] Code splitting with Next.js
- [x] Lazy loading for routes
- [x] Optimized animations (GPU-accelerated)
- [x] Reduced motion support
- [x] Image optimization ready
- [x] Bundle size optimized
- [ ] Add Lighthouse CI
- [ ] Add performance monitoring

---

## Deployment Options

### Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

### Netlify
```bash
npm run build
# Upload .next folder
```

### Self-Hosted
```bash
npm run build
npm run start
# Runs on port 3000
```

---

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Framer Motion](https://www.framer.com/motion/)
- [Radix UI](https://www.radix-ui.com/)
- [D3.js](https://d3js.org/)

---

## Need Help?

- Check the README.md for feature documentation
- Review existing components for patterns
- Refer to the transformation plan in the root directory

---

**Happy coding! 🚀**
