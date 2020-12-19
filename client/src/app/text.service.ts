import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class TextService {

  constructor(private http: HttpClient) { }

  submitText(text: any) {
    let url =  'http://127.0.0.1:8080/process_text/' + encodeURIComponent(text);
    return this.http.get(url);
  }

}
