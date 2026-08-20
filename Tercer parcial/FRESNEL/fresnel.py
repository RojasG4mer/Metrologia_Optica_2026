import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter

# %% Función de reconstrucción de Fresnel
def reconstruir(holo, d, lam, d_xi, d_eta):
    # Tamaño del holograma
    M, N = holo.shape

    # Resoluciones del plano reconstruido
    d_x = (d * lam) / (N * d_xi)
    d_y = (d * lam) / (M * d_eta)

    # Índices centrados
    i_M = np.arange(-M//2, M//2)
    i_N = np.arange(-N//2, N//2)

    # Malla del plano del holograma
    xi, eta = np.meshgrid(i_N * d_xi, i_M * d_eta)

    # Ejes del plano reconstruido
    x = i_N * d_x
    y = i_M * d_y

    # Reconstrucción (Propagación)
    U = np.exp(1j * np.pi * (xi**2 + eta**2) / (d * lam))
    U = U * holo
    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U))) 
    
    return U, x, y

# %% Función auxiliar para generar las 3 imágenes por objeto
def mostrar_resultados(holo, U, x_rec, y_rec, d_xi, d_eta, d_z, titulo, p_min, p_max):
    M, N = holo.shape
    eps = np.finfo(float).eps
    
    # 1. Ejes del holograma en milímetros [mm]
    x_holo = np.arange(-N//2, N//2) * d_xi * 1e3
    y_holo = np.arange(-M//2, M//2) * d_eta * 1e3
    extent_holo = [x_holo[0], x_holo[-1], y_holo[-1], y_holo[0]]
    
    # 2. Transformada de Fourier 2D del Holograma (Intensidad en log10)
    FFT_holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(holo)))
    L_fft = np.log10(np.abs(FFT_holo)**2 + eps)
    
    # Ejes de la Transformada en frecuencias espaciales [mm^-1]
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=d_xi)) / 1e3 
    fy = np.fft.fftshift(np.fft.fftfreq(M, d=d_eta)) / 1e3
    extent_fft = [fx[0], fx[-1], fy[-1], fy[0]]
    
    # 3. Intensidad Reconstruida [mm]
    I_rec = np.abs(U)**2
    L_rec = np.log10(I_rec + eps)
    extent_rec = [x_rec[0]*1e3, x_rec[-1]*1e3, y_rec[-1]*1e3, y_rec[0]*1e3]
    
    # Crear la figura para este objeto
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{titulo} (d = {d_z*1e3:.1f} mm)', fontsize=16, fontweight='bold')
    
    # Subplot 1: Holograma
    im0 = axes[0].imshow(holo, cmap='gray', extent=extent_holo)
    axes[0].set_title('Holograma Original')
    axes[0].set_xlabel('x [mm]')
    axes[0].set_ylabel('y [mm]')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Subplot 2: Transformada de Fourier
    # Ajustamos percentiles entre 50 y 99.9 para que destaquen los órdenes de difracción (DC, +1, -1)
    vmin_fft, vmax_fft = np.percentile(L_fft, [50, 99.9]) 
    im1 = axes[1].imshow(L_fft, cmap='gray', extent=extent_fft, vmin=vmin_fft, vmax=vmax_fft)
    axes[1].set_title('Transformada de Fourier')
    axes[1].set_xlabel('f_x [mm⁻¹]')
    axes[1].set_ylabel('f_y [mm⁻¹]')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Subplot 3: Intensidad Reconstruida
    vmin_rec, vmax_rec = np.percentile(L_rec, [p_min, p_max])
    im2 = axes[2].imshow(L_rec, cmap='gray', extent=extent_rec, vmin=vmin_rec, vmax=vmax_rec)
    axes[2].set_title('Intensidad Reconstruida')
    axes[2].set_xlabel('x [mm]')
    axes[2].set_ylabel('y [mm]')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.show()

# %% Parámetros de Entrada
d_xi = 6.45e-6   # Tamaño píxel horizontal
d_eta = 6.45e-6  # Tamaño píxel vertical
lam = 555e-9     # Láser verde
d_prueba = 0.850 # Distancia objeto de prueba
d_pequeno = 0.115 # Dado pequeño
d_grande = 0.200 # Dado grande

# %% Carga de imágenes
HOLO_ref = io.imread('Tercer parcial\FRESNEL\holograma900mm.tif', as_gray=True).astype(np.float64)
HOLO_peq = io.imread('Tercer parcial\FRESNEL\Dadopequenochido_00001.tif', as_gray=True).astype(np.float64)
HOLO_gra = io.imread('Tercer parcial\FRESNEL\Dadograndechido_00009.tif', as_gray=True).astype(np.float64)

# %% Filtrado Gaussiano
HOLO_ref = HOLO_ref - gaussian_filter(HOLO_ref, sigma=3)
HOLO_peq = HOLO_peq - gaussian_filter(HOLO_peq, sigma=1)
HOLO_gra = HOLO_gra - gaussian_filter(HOLO_gra, sigma=1)

# %% Llamada a la función de reconstrucción
U_ref, x_ref, y_ref = reconstruir(HOLO_ref, d_prueba, lam, d_xi, d_eta)
U_peq, x_peq, y_peq = reconstruir(HOLO_peq, d_pequeno, lam, d_xi, d_eta)
U_gra, x_gra, y_gra = reconstruir(HOLO_gra, d_grande, lam, d_xi, d_eta)

# %% Graficado de las 3 ventanas separadas (Holograma, Fourier e Intensidad)
# Objeto de referencia
mostrar_resultados(HOLO_ref, U_ref, x_ref, y_ref, d_xi, d_eta, d_prueba, 
                   titulo='Objeto de referencia', p_min=20, p_max=100)

# Dado pequeño
mostrar_resultados(HOLO_peq, U_peq, x_peq, y_peq, d_xi, d_eta, d_pequeno, 
                   titulo='Dado pequeño', p_min=60, p_max=99)

# Dado grande
mostrar_resultados(HOLO_gra, U_gra, x_gra, y_gra, d_xi, d_eta, d_grande, 
                   titulo='Dado grande', p_min=65, p_max=99)