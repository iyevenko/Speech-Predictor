import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {API_URL} from './env';

@Injectable({
  providedIn: 'root'
})
export class TextService {

  constructor(private http: HttpClient) { }

  submitText(text: any) {
    let url = `${API_URL}/rest/process-text/` + encodeURIComponent(text);
    return this.http.get(url);
  }

}
