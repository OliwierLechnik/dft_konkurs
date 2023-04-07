import numpy as np
import math
import pygame
import time
import torch
import librosa

pygame.init()


class Chart:
    def __init__(self, origin: tuple[int, int], size: tuple[int, int], scale: tuple[float, float],
                 color: tuple[int, int, int], surface):
        self.origin = origin
        self.sz = size
        self.scale = scale
        self.points = []
        self.surface = surface
        self.color = color

    def plot(self, x: list[float], y: list[float]):

        sy = self.sz[1] // 2
        sx = self.sz[0]

        y = list(map(lambda z: sy - z / self.scale[1] * sy + self.origin[1], y))
        x = list(map(lambda z: z / self.scale[0] * sx + self.origin[0], x))
        self.points = list(zip(x, y))

    def plot_tuple(self, xy: list[float, float]):
        sy = self.sz[1] // 2
        sx = self.sz[0]

        y = list(map(lambda z: sy - z[1] / self.scale[1] * sy + self.origin[1], xy))
        x = list(map(lambda z: z[0] / self.scale[0] * sx + self.origin[0], xy))
        self.points = list(zip(x, y))

    def showline(self):
        for i in range(1, len(self.points)):
            pygame.draw.aaline(self.surface, self.color, self.points[i - 1], self.points[i], 4)

    def showbar(self):
        for x, y in self.points:
            pygame.draw.line(self.surface, self.color, (x, self.origin[1] + self.sz[1] // 2), (x, y), 2)


def prepper_points(desired_size: int, data, style=lambda x: x):
    # it will group data for pretier chart
    def avg(slice):
        if len(slice) == 0:
            return 0
        return sum(slice) / len(slice)

    sz = len(data)
    real_dencity = sz / desired_size
    int_dencity = math.ceil(real_dencity)

    return [(i * int_dencity / real_dencity, avg(data[i * int_dencity:(i + 1) * int_dencity])) for i in
            range(desired_size - 1)]


def ecos(f, t):
    return math.cos(2 * math.pi * f * t)


def esin(f, t):
    return math.sin(2 * math.pi * f * t)


def unit_matrix(freq, bitrate, window_duration):
    def unit_samples(f, rate, duration):
        return [
            [ecos(f, i / rate) for i in range(int(rate * duration))],
            [esin(f, i / rate) for i in range(int(rate * duration))]
        ]

    out1 = []
    out2 = []
    for i in range(freq - 1):
        unit = unit_samples(i, bitrate, window_duration)
        out1.append(unit[0])
        out2.append(unit[1])
    return [
        np.array(out1),
        np.array(out2)
    ]

def unit_matrix_gpu(freq, bitrate, window_duration, device):
    def unit_samples(f, rate, duration):
        t = torch.arange(int(rate * duration), device=device).float() / rate
        return [
            torch.cos(2 * torch.pi * f * t).float(),
            torch.sin(2 * torch.pi * f * t).float()
        ]

    out1 = []
    out2 = []
    for i in range(freq - 1):
        unit = unit_samples(i, bitrate, window_duration)
        out1.append(unit[0])
        out2.append(unit[1])
    return [
        torch.stack(out1),
        torch.stack(out2)
    ]


def abs_(x):
    return x if x > 0 else -1 * x


def transform_cpu(matrix, wave):
    x = np.dot(matrix[0], wave)
    y = np.dot(matrix[1], wave)
    n = len(wave)
    return np.sqrt(x**2+y**2) / n

def transform_gpu(matrix, wave, device):

    wave = torch.tensor(wave, device=device).float()
    x = torch.matmul(matrix[0], wave)
    y = torch.matmul(matrix[1], wave)
    return (torch.sqrt(x**2 + y**2)).cpu().data.numpy()


def main():
    W = 1600
    H = 400


    # bitrate = 22050
    window_duration = .15
    freqs = 4000
    path = 'elise2.wav'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp, bitrate = librosa.load(path)
    data = np.append(np.array([0] * int(bitrate * window_duration)), tmp)
    # print(data)

    window_shape = np.array([
        esin(0.5, i / int(bitrate * window_duration)) ** 1.5 for i in range(int(bitrate * window_duration))
    ])

    visualiser_shape = np.array([
        i ** 1.085 / freqs + 0.5 for i in range(freqs - 1)
    ])

    pygame.display.set_caption('DFT by Oliwier Lechnik')
    window = pygame.display.set_mode((W, H * 2))

    running = True

    time_domain = Chart((0, 0,), (W, H), (W, 1.2), (int('fb',16), int('f1',16), int('c7',16)), window)
    window_shape_chart = Chart((0, 0,), (W, H), (W, 1.2), (255, 255, 255), window)
    freq_domain = Chart((0, H // 2 * 3 - 20,), (W, H), (W, 200), (int('fb',16), int('f1',16), int('c7',16)), window)
    visualiser = Chart((0, H // 2 * 3 - 20,), (W, H), (freqs, 1), (255, 255, 255), window)

    # matrix = unit_matrix(freqs, bitrate, window_duration)
    # matrix = torch.tensor(matrix, device=device)
    matrix = unit_matrix_gpu(freqs,bitrate,window_duration,device)


    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    t = time.time()
    f = 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        d = time.time()
        dt = time.time() - t
        # print(dt)
        start = int(dt * bitrate)
        end = start + int(window_duration * bitrate)
        if end > len(data):
            pygame.quit()
        wave = data[start:end]

        windowed_wave = np.multiply(wave, window_shape)

        wave_points = prepper_points(W, np.convolve(windowed_wave, np.repeat(1,15)/15))

        fourier = transform_gpu(matrix, wave, device)
        fourier_graph = np.multiply(visualiser_shape, fourier)
        smooth_fourier_graph = np.convolve(fourier_graph, np.repeat(1,15)/15)
        fourier_graph_points = prepper_points(W,smooth_fourier_graph)
        # print(fourier)

        window.fill(
            (int('28',16),
            int('28',16),
            int('28',16))
        )

        time_domain.plot_tuple(wave_points)
        time_domain.showline()

        freq_domain.plot_tuple(fourier_graph_points)
        freq_domain.showline()

        pygame.display.flip()
        # print(1/(time.time() - d),'fps')
        print(list(fourier).index(max(fourier)))



if __name__ == '__main__':
    main()
