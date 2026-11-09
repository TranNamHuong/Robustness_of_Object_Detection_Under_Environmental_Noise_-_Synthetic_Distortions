import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';

afterEach(() => { cleanup(); window.localStorage.clear(); vi.unstubAllGlobals(); });

describe('research lab routes', () => {
  it.each([
    ['/', 'Phát hiện vật thể'],
    ['/comparison', 'So sánh & đề xuất'],
    ['/analytics', 'Phân tích nghiên cứu'],
    ['/methodology', 'Phương pháp & thư viện'],
  ])('renders %s without invented benchmark results', (path, heading) => {
    render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
    expect(screen.getByRole('heading', { level: 1, name: heading })).toBeInTheDocument();
    if (path === '/comparison') {
      expect(screen.getByLabelText(/Điều kiện ảnh/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Trạng thái ảnh/)).toBeInTheDocument();
    }
    if (path === '/analytics') expect(screen.getByLabelText(/Loại biến dạng/)).toBeInTheDocument();
    if (path === '/methodology') expect(screen.getByText(/Quy trình thực nghiệm/)).toBeInTheDocument();
  });

  it('explains why unlabeled image uploads do not get accuracy metrics', () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(screen.getByText(/Ảnh tải lên không có nhãn nên không hiển thị mAP/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Phát hiện/ })).toBeDisabled();
  });

  it('opens the bilingual user guide from the detection page', () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: 'Mở hướng dẫn' }));
    expect(screen.getByRole('dialog', { name: /Cách dùng Detection Lab/ })).toBeInTheDocument();
    expect(screen.getByText(/Tải ảnh JPG, PNG hoặc WebP/)).toBeInTheDocument();
  });

  it('switches to the dark theme and provides bilingual measured conditions', () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: /Bật giao diện tối/ }));
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
    expect(screen.getByRole('option', { name: /Nhiễu Gaussian/ })).toBeInTheDocument();
  });

  it('switches all page copy with the selected language', () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    fireEvent.change(screen.getByRole('combobox', { name: /Interface language/ }), { target: { value: 'en' } });
    expect(screen.getByRole('heading', { level: 1, name: 'Object detection' })).toBeInTheDocument();
    expect(screen.getByText('Comparison')).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: /Interface language/ }), { target: { value: 'vi' } });
    expect(screen.getByRole('heading', { level: 1, name: 'Phát hiện vật thể' })).toBeInTheDocument();
  });

  it('opens the source image in a full-screen lightbox', () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    const input = screen.getByLabelText('Ảnh đầu vào') as HTMLInputElement;
    const file = new File(['image'], 'room.png', { type: 'image/png' });
    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByRole('button', { name: 'Bấm để phóng to Ảnh gốc' }));
    expect(screen.getByRole('dialog', { name: 'Ảnh phóng to' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Đóng ảnh lớn' }));
    expect(screen.queryByRole('dialog', { name: 'Ảnh phóng to' })).not.toBeInTheDocument();
  });

  it('smoothly navigates the datasets link to the provenance section', async () => {
    const scrollSpy = vi.fn();
    vi.stubGlobal('scrollTo', scrollSpy);
    render(<MemoryRouter><App /></MemoryRouter>);
    const datasetsLink = screen.getByRole('link', { name: 'Dữ liệu & nguồn' });
    expect(datasetsLink).toHaveAttribute('href', '/methodology#provenance');
    fireEvent.click(datasetsLink);
    expect(document.getElementById('provenance')).toBeInTheDocument();
    await waitFor(() => expect(scrollSpy).toHaveBeenCalledWith(expect.objectContaining({ behavior: 'smooth' })));
  });
});
