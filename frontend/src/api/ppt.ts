import apiClient from './client'
import type { PPTDeck } from '@/types/ppt'

export const pptApi = {
  getDecks: async (): Promise<PPTDeck[]> => {
    const response = await apiClient.get<PPTDeck[]>('/ppt/decks')
    return response.data
  },

  getDeck: async (id: string): Promise<PPTDeck> => {
    const response = await apiClient.get<PPTDeck>(`/ppt/decks/${id}`)
    return response.data
  },

  createDeck: async (deck: Partial<PPTDeck>): Promise<PPTDeck> => {
    const response = await apiClient.post<PPTDeck>('/ppt/decks', deck)
    return response.data
  },

  updateDeck: async (id: string, deck: Partial<PPTDeck>): Promise<PPTDeck> => {
    const response = await apiClient.put<PPTDeck>(`/ppt/decks/${id}`, deck)
    return response.data
  },

  deleteDeck: async (id: string): Promise<void> => {
    await apiClient.delete(`/ppt/decks/${id}`)
  },
}
